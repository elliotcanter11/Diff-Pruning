import huggingface_hub
from huggingface_hub import constants as hf_constants

if not hasattr(hf_constants, "hf_cache_home"):
    hf_constants.hf_cache_home = hf_constants.HF_HUB_CACHE

if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

if not hasattr(huggingface_hub, "HfFolder"):
    class HfFolder:
        @staticmethod
        def get_token():
            return huggingface_hub.get_token()
    huggingface_hub.HfFolder = HfFolder

import jax
if not hasattr(jax.random, "KeyArray"):
    jax.random.KeyArray = jax.Array

import transformers.utils as tf_utils
if not hasattr(tf_utils, "FLAX_WEIGHTS_NAME"):
    tf_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"

from diffusers import DiffusionPipeline, DDPMPipeline, DDIMPipeline, DDIMScheduler, DDPMScheduler

from diffusers import DiffusionPipeline, DDPMPipeline, DDIMPipeline, DDIMScheduler, DDPMScheduler
from diffusers.models import UNet2DModel
import torch_pruning as tp
import torch
import torchvision
from torchvision import transforms as T
import torchvision
from tqdm import tqdm
import os
from glob import glob
from PIL import Image
import accelerate
import utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,  default=None, help="path to an image folder")
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--pruning_ratio", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--pruner", type=str, default='taylor', choices=['taylor', 'random', 'magnitude', 'reinit', 'diff-pruning', 'mi'])

parser.add_argument("--thr", type=float, default=0.05, help="threshold for diff-pruning")

# MI pruner (closed-form Gaussian conditional MI). Set a weight to 0 to disable that term.
parser.add_argument("--mi_w_output", type=float, default=1.0, help="weight of the output-MI term (channel vs final predicted noise)")
parser.add_argument("--mi_w_adjacency", type=float, default=1.0, help="weight of the adjacency-MI term (channel vs next-layer activations)")
parser.add_argument("--mi_w_mid", type=float, default=1.0, help="weight of the mid-range-MI term (channel vs a layer halfway to the output)")
parser.add_argument("--mi_mid_alpha", type=float, default=0.5, help="how far downstream the mid-range target sits, as a fraction of the remaining depth")
parser.add_argument("--mi_num_batches", type=int, default=32, help="number of calibration forward passes (each = batch_size images) for the MI pruner")
parser.add_argument("--mi_num_locations", type=int, default=4, help="spatial locations sampled per image for the adjacency and mid-range terms")
parser.add_argument("--mi_out_grid", type=int, default=1, help="per-channel gxg pooled descriptor for the whole-layer output term (1=well-conditioned; 2 keeps spatial but needs ~3x more images)")
parser.add_argument("--mi_out_target_pool", type=int, default=8, help="pooled grid of the output target for the output term")
parser.add_argument("--mi_shrinkage", type=float, default=1e-2, help="ridge shrinkage on the covariance for the Gaussian MI estimate")

args = parser.parse_args()

batch_size = args.batch_size
dataset = args.dataset

if __name__=='__main__':

    cifar_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=0.5, std=0.5),
    ])
    
    # loading images for gradient-based / activation-based pruning
    if args.pruner in ['taylor', 'diff-pruning', 'mi']:
        dataset = utils.get_dataset(args.dataset, transform=cifar_transform)
        print(f"Dataset size: {len(dataset)}")
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
        )
        import torch_pruning as tp
        clean_images = next(iter(train_dataloader))
        if isinstance(clean_images, (list, tuple)):
            clean_images = clean_images[0]
        clean_images = clean_images.to(args.device)
        noise = torch.randn(clean_images.shape).to(clean_images.device)

    # Loading pretrained model
    print("Loading pretrained model from {}".format(args.model_path))
    pipeline = DDPMPipeline.from_pretrained(args.model_path).to(args.device)
    scheduler = pipeline.scheduler
    model = pipeline.unet.eval()
    if 'cifar' in args.model_path:
        example_inputs = {'sample': torch.randn(1, 3, 32, 32).to(args.device), 'timestep': torch.ones((1,)).long().to(args.device)}
    else:
        example_inputs = {'sample': torch.randn(1, 3, 256, 256).to(args.device), 'timestep': torch.ones((1,)).long().to(args.device)}

    if args.pruning_ratio>0:
        if args.pruner == 'taylor':
            imp = tp.importance.TaylorImportance(multivariable=True) # standard first-order taylor expansion
        elif args.pruner == 'random' or args.pruner=='reinit':
            imp = tp.importance.RandomImportance()
        elif args.pruner == 'magnitude':
            imp = tp.importance.MagnitudeImportance()
        elif args.pruner == 'diff-pruning':
            imp = tp.importance.TaylorImportance(multivariable=False) # a modified version, estimating the accumulated error of weight removal
        elif args.pruner == 'mi':
            from mi_importance import MIImportance
            imp = MIImportance(
                w_output=args.mi_w_output,
                w_adjacency=args.mi_w_adjacency,
                w_mid=args.mi_w_mid,
                mid_alpha=args.mi_mid_alpha,
                num_locations=args.mi_num_locations,
                out_grid=args.mi_out_grid,
                out_target_pool=args.mi_out_target_pool,
                shrinkage=args.mi_shrinkage,
                prune_ratio=args.pruning_ratio,
            )
        else:
            raise NotImplementedError

        ignored_layers = [model.conv_out]
        channel_groups = {}
        from diffusers.models.attention_processor import Attention
        for m in model.modules():
            if isinstance(m, Attention):
                channel_groups[m.to_q] = m.heads
                channel_groups[m.to_k] = m.heads
                channel_groups[m.to_v] = m.heads
        
        # one-shot prune; the MI importance does its own greedy (redundancy-aware)
        # elimination internally, so no iterative_steps are needed here.
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=1,
            channel_groups=channel_groups,
            pruning_ratio=args.pruning_ratio,
            ignored_layers=ignored_layers,
        )

        base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
        model.zero_grad()
        model.eval()
        import random

        if args.pruner in ['taylor', 'diff-pruning']:
            loss_max = 0
            print("Accumulating gradients for pruning...")
            for step_k in tqdm(range(1000)):
                timesteps = (step_k*torch.ones((args.batch_size,), device=clean_images.device)).long()
                noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
                model_output = model(noisy_images, timesteps).sample
                loss = torch.nn.functional.mse_loss(model_output, noise) 
                loss.backward() 
                
                if args.pruner=='diff-pruning':
                    if loss>loss_max: loss_max = loss
                    if loss<loss_max * args.thr: break # taylor expansion over pruned timesteps ( L_t / L_max > thr )

        def mi_calibrate():
            # collect activations on the CURRENT (possibly partially pruned) model,
            # so MI is re-estimated on the surviving channels each greedy step.
            imp.reset()
            imp.attach(model, ignored_layers)
            num_train_timesteps = scheduler.config.num_train_timesteps
            mi_iter = iter(train_dataloader)
            with torch.no_grad():
                for _ in range(args.mi_num_batches):
                    try:
                        batch = next(mi_iter)
                    except StopIteration:
                        mi_iter = iter(train_dataloader)
                        batch = next(mi_iter)
                    if isinstance(batch, (list, tuple)):
                        batch = batch[0]
                    batch = batch.to(args.device)
                    timesteps = torch.randint(
                        0, num_train_timesteps, (batch.shape[0],), device=batch.device
                    ).long()
                    step_noise = torch.randn_like(batch)
                    noisy_images = scheduler.add_noise(batch, step_noise, timesteps)
                    imp.new_pass(batch.shape[0])
                    model_output = model(noisy_images, timesteps).sample
                    imp.record_output(model_output)
                    imp.record_timesteps(timesteps)
            imp.finalize()

        if args.pruner == 'mi':
            print("Collecting activations for MI-based pruning...")
            mi_calibrate()

        for g in pruner.step(interactive=True):
            g.prune()

        if args.pruner == 'mi':
            imp.report_diagnostic()

        # Update static attributes
        from diffusers.models.resnet import Upsample2D, Downsample2D
        for m in model.modules():
            if isinstance(m, (Upsample2D, Downsample2D)):
                m.channels = m.conv.in_channels
                m.out_channels = m.conv.out_channels

        macs, params = tp.utils.count_ops_and_params(model, example_inputs)
        print(model)
        print("#Params: {:.4f} M => {:.4f} M".format(base_params/1e6, params/1e6))
        print("#MACS: {:.4f} G => {:.4f} G".format(base_macs/1e9, macs/1e9))
        model.zero_grad()
        del pruner

        if args.pruner=='reinit':
            def reset_parameters(model):
                for m in model.modules():
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
            reset_parameters(model)

    pipeline.save_pretrained(args.save_path)
    if args.pruning_ratio>0:
        os.makedirs(os.path.join(args.save_path, "pruned"), exist_ok=True)
        torch.save(model, os.path.join(args.save_path, "pruned", "unet_pruned.pth"))

    # Sampling images from the pruned model
    pipeline = DDIMPipeline(
        unet = model,
        scheduler = DDIMScheduler.from_pretrained(args.save_path, subfolder="scheduler")
    )
    with torch.no_grad():
        generator = torch.Generator(device=pipeline.device).manual_seed(0)
        pipeline.to("cuda")
        images = pipeline(num_inference_steps=100, batch_size=args.batch_size, generator=generator, output_type="numpy").images
        os.makedirs(os.path.join(args.save_path, 'vis'), exist_ok=True)
        torchvision.utils.save_image(torch.from_numpy(images).permute([0, 3, 1, 2]), "{}/vis/after_pruning.png".format(args.save_path))
        
