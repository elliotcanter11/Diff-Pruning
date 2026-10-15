#!/usr/bin/env python3
"""Unified CLI workflow for Diffusion Model Pruning and Evaluation.

Replicates and extends the workflow from `diffpruning4.ipynb` and `mi_vs_magnitude.ipynb`:
  1. Environment & data prerequisite checks (with optional --setup)
  2. Baseline pretrained reference sampling (cached across runs)
  3. Pruning: Magnitude, MI (adjacency, output, mid-range, combined)
  4. Finetuning (reproducible seed, mixed-precision fp16, EMA)
  5. Deterministic paired sampling (matched seeds for direct SSIM)
  6. Evaluation: SSIM vs original model & optional FID score
  7. Formatted results table + JSON output

Usage:
  # Quick comparison on GPU with 1000 finetune iterations:
  python run_workflow.py --methods magnitude mi_combined --iters 1000 --ssim_samples 1000

  # Full 5-method comparison:
  python run_workflow.py --methods all --ratio 0.3 --iters 1000

  # First time setup on server (downloads dataset + converts pretrained checkpoint):
  python run_workflow.py --setup

  # Single method run:
  python run_workflow.py --methods mi_combined --ratio 0.2 --iters 5000
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time

# Ensure child commands use this python environment and find the local diffusers package
_repo_root = os.path.dirname(os.path.abspath(__file__))
_py_bin_dir = os.path.dirname(os.path.abspath(sys.executable))
if not os.environ.get("PATH", "").startswith(_py_bin_dir):
    os.environ["PATH"] = f"{_py_bin_dir}{os.path.pathsep}{os.environ.get('PATH', '')}"
_cur_ppath = os.environ.get("PYTHONPATH", "")
if _repo_root not in _cur_ppath.split(os.path.pathsep):
    os.environ["PYTHONPATH"] = f"{_repo_root}{os.path.pathsep}{_cur_ppath}" if _cur_ppath else _repo_root


ALL_METHODS = ["magnitude", "mi_adjacency", "mi_output", "mi_mid", "mi_combined"]

METHOD_CONFIGS = {
    "magnitude": {
        "pruner": "magnitude",
        "w_out": 0.0,
        "w_adj": 0.0,
        "w_mid": 0.0,
        "desc": "Standard magnitude pruning baseline",
    },
    "mi_adjacency": {
        "pruner": "mi",
        "w_out": 0.0,
        "w_adj": 1.0,
        "w_mid": 0.0,
        "desc": "MI adjacency term only (channel vs next-layer)",
    },
    "mi_output": {
        "pruner": "mi",
        "w_out": 1.0,
        "w_adj": 0.0,
        "w_mid": 0.0,
        "desc": "MI output term only (channel vs predicted noise)",
    },
    "mi_mid": {
        "pruner": "mi",
        "w_out": 0.0,
        "w_adj": 0.0,
        "w_mid": 1.0,
        "desc": "MI mid-range term only (channel vs halfway layer)",
    },
    "mi_combined": {
        "pruner": "mi",
        "w_out": 1.0,
        "w_adj": 1.0,
        "w_mid": 1.0,
        "desc": "Multiscale MI ladder (adjacency + mid-range + output)",
    },
}


def log(msg, bold=False):
    prefix = "\033[1m" if bold else ""
    suffix = "\033[0m" if bold else ""
    print(f"{prefix}{msg}{suffix}", flush=True)


def run_command(cmd, desc=None, stream=True, capture=False):
    """Run shell command with optional streaming or output capture."""
    if desc:
        log(f"\n--> {desc}...", bold=True)
    log(f"$ {cmd}")
    
    if capture:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if p.returncode != 0:
            log(f"Command failed with code {p.returncode}: {cmd}", bold=True)
            if p.stderr:
                print("STDERR:\n" + p.stderr[-2000:])
            if p.stdout:
                print("STDOUT:\n" + p.stdout[-2000:])
            raise RuntimeError(f"Command failed: {cmd}")
        return p.stdout
    else:
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError(f"Command failed (exit code {ret}): {cmd}")
        return ""


def parse_metric(pattern, text, default=float("nan")):
    m = re.search(pattern, text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return default
    return default


def ensure_setup(args):
    """Verify or download/prepare datasets and pretrained weights."""
    log("\nChecking prerequisites...", bold=True)

    # 1. Dataset check
    dataset_exists = os.path.isdir(args.dataset) and len(os.listdir(args.dataset)) > 100
    if not dataset_exists:
        if args.setup:
            log(f"Dataset not found at {args.dataset}. Extracting CIFAR-10...")
            os.makedirs("data", exist_ok=True)
            try:
                run_command("python tools/extract_cifar10_hug.py --output data", desc="Extracting CIFAR-10 via Hugging Face")
            except Exception as e:
                log(f"Hugging Face extraction failed ({e}), falling back to torchvision...")
                run_command("python tools/extract_cifar10.py --output data", desc="Extracting CIFAR-10 via torchvision")
        else:
            log(f"WARNING: Dataset not found at '{args.dataset}'.", bold=True)
            log("Run with `--setup` or run: python tools/extract_cifar10_hug.py --output data")
            sys.exit(1)
    else:
        log(f"  [OK] Dataset found at {args.dataset}")

    # 2. Pretrained model check
    model_exists = os.path.isdir(args.model_path) and os.path.exists(os.path.join(args.model_path, "unet"))
    if not model_exists:
        if args.setup:
            log(f"Pretrained model not found at {args.model_path}. Converting checkpoint...")
            run_command("bash tools/convert_cifar10_ddpm_ema.sh", desc="Downloading and converting CIFAR-10 DDPM checkpoint")
        else:
            log(f"WARNING: Pretrained model not found at '{args.model_path}'.", bold=True)
            log("Run with `--setup` or run: bash tools/convert_cifar10_ddpm_ema.sh")
            sys.exit(1)
    else:
        log(f"  [OK] Pretrained model found at {args.model_path}")

    # 3. FID stats check (if FID is requested)
    fid_stats_path = os.path.join(args.run_dir, "fid_stats_cifar10.npz")
    if args.compute_fid:
        if not os.path.exists(fid_stats_path):
            log(f"FID stats not found at {fid_stats_path}. Precomputing stats...")
            os.makedirs(args.run_dir, exist_ok=True)
            run_command(
                f"python fid_score.py --save-stats {args.dataset} {fid_stats_path} "
                f"--device {args.device} --batch-size {args.batch_size}",
                desc="Precomputing CIFAR-10 FID statistics",
            )
        else:
            log(f"  [OK] FID stats found at {fid_stats_path}")


def count_images_in_dir(path):
    if not os.path.exists(path):
        return 0
    return len(glob.glob(os.path.join(path, "**/*.png"), recursive=True))


def ensure_baseline_samples(args):
    """Ensure baseline pretrained model has reference samples for SSIM & FID."""
    baseline_dir = os.path.join(args.run_dir, "sample", "ddpm_cifar10_pretrained")
    target_count = args.fid_samples if args.compute_fid else args.ssim_samples
    existing_count = count_images_in_dir(baseline_dir)

    if existing_count >= target_count and not args.force_resample_baseline:
        log(f"\n[OK] Baseline pretrained samples already exist ({existing_count} images at {baseline_dir}). Skipping sampling.")
        return baseline_dir

    log(f"\nGenerating {target_count} reference samples from pretrained baseline...", bold=True)
    os.makedirs(baseline_dir, exist_ok=True)
    run_command(
        f"python ddpm_sample.py --output_dir {baseline_dir} "
        f"--batch_size {args.batch_size} --total_samples {target_count} "
        f"--ddim_steps {args.ddim_steps} --model_path {args.model_path} "
        f"--skip_type uniform --seed 0",
        desc="Sampling pretrained reference images",
    )
    return baseline_dir


def run_pruning(method_name, cfg, args, save_path):
    """Run ddpm_prune.py and parse output metrics."""
    os.makedirs(save_path, exist_ok=True)
    prune_bs = 256 if cfg["pruner"] == "mi" else 128

    cmd = (
        f"python ddpm_prune.py "
        f"--dataset {args.dataset} "
        f"--model_path {args.model_path} "
        f"--save_path {save_path} "
        f"--pruning_ratio {args.ratio} "
        f"--batch_size {prune_bs} "
        f"--pruner {cfg['pruner']} "
        f"--device {args.device}"
    )
    if cfg["pruner"] == "mi":
        cmd += (
            f" --mi_w_output {cfg['w_out']}"
            f" --mi_w_adjacency {cfg['w_adj']}"
            f" --mi_w_mid {cfg['w_mid']}"
            f" --mi_num_batches {args.mi_num_batches}"
            f" --mi_num_locations {args.mi_num_locations}"
        )

    out = run_command(cmd, desc=f"Pruning with {method_name} ({cfg['desc']})", capture=True)
    
    # Print the summary portions of stdout (params, macs, diagnostic)
    for line in out.splitlines():
        if any(keyword in line for keyword in ["#Params:", "#MACS:", "[MIImportance]", "Jaccard", "Spearman"]):
            print(line)

    params = parse_metric(r"#Params:.*=>\s*([\d.]+)\s*M", out)
    macs = parse_metric(r"#MACS:.*=>\s*([\d.]+)\s*G", out)
    jaccard = parse_metric(r"Jaccard\s*=\s*([\d.]+)", out, default=1.0 if cfg["pruner"] == "magnitude" else float("nan"))
    spearman = parse_metric(r"Spearman rank corr\s*=\s*([\-\d.]+)", out, default=1.0 if cfg["pruner"] == "magnitude" else float("nan"))

    return {
        "params": params,
        "macs": macs,
        "jaccard": jaccard,
        "spearman": spearman,
    }


def run_finetuning(method_name, prune_dir, finetune_dir, args):
    """Run ddpm_train.py to finetune the pruned network."""
    os.makedirs(finetune_dir, exist_ok=True)
    save_steps = min(1000, args.iters)
    cmd = (
        f"python ddpm_train.py "
        f"--dataset={args.dataset} "
        f"--model_path={prune_dir} "
        f"--pruned_model_ckpt={prune_dir}/pruned/unet_pruned.pth "
        f"--resolution=32 "
        f"--output_dir={finetune_dir} "
        f"--train_batch_size={args.train_batch_size} "
        f"--num_iters={args.iters} "
        f"--gradient_accumulation_steps=1 "
        f"--learning_rate=2e-4 "
        f"--lr_warmup_steps=0 "
        f"--save_model_steps={save_steps} "
        f"--dataloader_num_workers={args.workers} "
        f"--adam_weight_decay=0.00 "
        f"--dropout=0.1 "
        f"--mixed_precision=fp16"
    )
    if args.use_ema:
        cmd += " --use_ema --ema_max_decay=0.9999"
    run_command(cmd, desc=f"Finetuning {method_name} ({args.iters} iterations, EMA={'on' if args.use_ema else 'off'})", stream=True)


def find_final_checkpoint(finetune_dir, prune_dir, use_ema=False):
    """Find the final checkpoint from finetuning (default: unet_pruned.pth)."""
    pruned_dir = os.path.join(finetune_dir, "pruned")
    
    # 1. If EMA was explicitly requested, check for final EMA checkpoint
    if use_ema:
        ema_final = os.path.join(pruned_dir, "unet_ema_pruned.pth")
        if os.path.exists(ema_final):
            return ema_final, finetune_dir

    # 2. Check for the final standard checkpoint unet_pruned.pth
    final_ckpt = os.path.join(pruned_dir, "unet_pruned.pth")
    if os.path.exists(final_ckpt):
        return final_ckpt, finetune_dir

    # 3. Check for numbered step checkpoints (e.g. unet_pruned-<step>.pth) and pick the latest
    step_ckpts = glob.glob(os.path.join(pruned_dir, "unet_pruned-*.pth"))
    if step_ckpts:
        def get_step(path):
            m = re.search(r"unet_pruned-(\d+)\.pth", path)
            return int(m.group(1)) if m else -1
        latest_ckpt = max(step_ckpts, key=get_step)
        return latest_ckpt, finetune_dir

    # 4. Fallback to pre-finetuned pruned model
    raw_ckpt = os.path.join(prune_dir, "pruned", "unet_pruned.pth")
    if os.path.exists(raw_ckpt):
        return raw_ckpt, prune_dir

    raise FileNotFoundError(f"Could not find any checkpoint in {finetune_dir} or {prune_dir}")


def run_sampling(method_name, finetune_dir, prune_dir, sample_dir, n_samples, args):
    """Sample images from the final checkpoint of the finetuned (or raw) pruned model."""
    os.makedirs(sample_dir, exist_ok=True)
    
    ckpt, model_path = find_final_checkpoint(finetune_dir, prune_dir, use_ema=args.use_ema)
    log(f"Using final checkpoint: {ckpt}")

    cmd = (
        f"python ddpm_sample.py "
        f"--output_dir={sample_dir} "
        f"--batch_size={args.batch_size} "
        f"--total_samples={n_samples} "
        f"--ddim_steps={args.ddim_steps} "
        f"--pruned_model_ckpt={ckpt} "
        f"--model_path={model_path} "
        f"--skip_type=uniform "
        f"--seed=0"
    )
    run_command(cmd, desc=f"Sampling {n_samples} images for {method_name}", stream=True)


def evaluate_ssim(sample_dir, baseline_dir):
    """Compute paired SSIM vs the pretrained reference model."""
    sample_p0 = os.path.join(sample_dir, "process_0")
    baseline_p0 = os.path.join(baseline_dir, "process_0")
    cmd = f"python ddpm_exp/compute_ssim.py --path {sample_p0} {baseline_p0}"
    out = run_command(cmd, desc="Computing SSIM vs Pretrained Baseline", capture=True)
    print(out.strip())
    ssim = parse_metric(r"ssim:\s*(?:tensor\()?([\-\d.]+)", out)
    return ssim


def evaluate_fid(sample_dir, fid_stats_path, args):
    """Compute FID score against CIFAR-10 stats."""
    cmd = (
        f"python fid_score.py {sample_dir} {fid_stats_path} "
        f"--device {args.device} --batch-size {args.batch_size}"
    )
    out = run_command(cmd, desc="Computing FID Score", capture=True)
    print(out.strip())
    fid = parse_metric(r"FID:\s*([\d.]+)", out)
    return fid


def print_results_table(results, args):
    """Render a clean comparison table matching notebook format."""
    cols = ["params", "macs", "ssim", "jaccard", "spearman"]
    if args.compute_fid:
        cols.append("fid")

    header_title = (
        f"ratio={args.ratio}  finetune={args.iters}  "
        f"ssim_samples={args.ssim_samples}  ema={'on' if args.use_ema else 'off'}  "
        f"fid={'on' if args.compute_fid else 'off'}"
    )
    notes = "SSIM higher = better (closer to original) | Jacc/Spear vs magnitude (low = different cuts)"

    sep = "=" * 80
    log("\n" + sep, bold=True)
    log(header_title, bold=True)
    log(notes)
    log("-" * 80)
    
    header_row = f"{'method':15s}" + "".join(f"{c:>10s}" for c in cols)
    log(header_row, bold=True)
    log("-" * 80)

    for name, r in results.items():
        row = f"{name:15s}"
        for c in cols:
            val = r.get(c, float("nan"))
            if c in ["params", "macs"]:
                row += f"{val:10.2f}" if not math.isnan(val) else f"{'nan':>10s}"
            elif c in ["ssim", "jaccard", "spearman"]:
                row += f"{val:10.4f}" if not math.isnan(val) else f"{'nan':>10s}"
            elif c == "fid":
                row += f"{val:10.2f}" if not math.isnan(val) else f"{'nan':>10s}"
            else:
                row += f"{val:10.3f}"
        print(row)
    log(sep + "\n", bold=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified CLI workflow for Diffusion Model Pruning and Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help=f"Methods to evaluate. Choose 'all' or subset of: {ALL_METHODS}",
    )
    parser.add_argument("--ratio", type=float, default=0.3, help="Pruning ratio")
    parser.add_argument("--iters", type=int, default=1000, help="Finetuning steps")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers for finetuning")
    parser.add_argument("--batch_size", type=int, default=256, help="Sampling & FID batch size")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Finetuning batch size")
    parser.add_argument("--ddim_steps", type=int, default=100, help="DDIM sampling steps")
    parser.add_argument("--ssim_samples", type=int, default=1000, help="Paired sample count for SSIM")
    parser.add_argument("--compute_fid", action="store_true", default=False, help="Compute FID score")
    parser.add_argument("--fid_samples", type=int, default=10000, help="Sample count when computing FID")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if subprocess.run("which nvidia-smi", shell=True, capture_output=True).returncode == 0 else "cpu",
        help="Device to use (e.g. cuda:0, cuda:1, cpu)",
    )
    parser.add_argument("--dataset", type=str, default="data/cifar10_images", help="Dataset directory")
    parser.add_argument("--model_path", type=str, default="pretrained/ddpm_ema_cifar10", help="Pretrained model dir")
    parser.add_argument("--run_dir", type=str, default="run", help="Base run output directory")
    parser.add_argument("--setup", action="store_true", help="Download CIFAR10 and convert pretrained model if missing")
    parser.add_argument("--skip_finetune", action="store_true", help="Skip finetuning (evaluates raw pruned model)")
    parser.add_argument("--force_resample_baseline", action="store_true", help="Force re-sampling pretrained baseline")
    parser.add_argument("--save_json", type=str, default=None, help="Path to save results JSON (default: <run_dir>/results.json)")
    parser.add_argument("--use_ema", action="store_true", default=False, help="Enable EMA during finetuning (default: False; uses final checkpoint)")

    # MI hyperparams
    parser.add_argument("--mi_num_batches", type=int, default=32, help="Number of calibration batches for MI")
    parser.add_argument("--mi_num_locations", type=int, default=4, help="Spatial locations sampled per image for MI")

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine methods to run
    if "all" in args.methods:
        methods = ALL_METHODS
    else:
        methods = [m for m in args.methods if m in METHOD_CONFIGS]
        if not methods:
            log(f"No valid methods specified in {args.methods}. Valid options: {ALL_METHODS}", bold=True)
            sys.exit(1)

    log("=" * 80, bold=True)
    log(" Diffusion Pruning & Evaluation Workflow ", bold=True)
    log("=" * 80, bold=True)
    log(f"Methods to run: {methods}")
    log(f"Pruning ratio : {args.ratio}")
    log(f"Finetune iters: {args.iters}")
    log(f"SSIM samples  : {args.ssim_samples}")
    log(f"Compute FID   : {args.compute_fid} (samples: {args.fid_samples if args.compute_fid else 'N/A'})")
    log(f"Use EMA       : {args.use_ema}")
    log(f"Device        : {args.device}")

    # Check / setup environment
    ensure_setup(args)

    # Ensure pretrained baseline samples are available
    baseline_dir = ensure_baseline_samples(args)

    n_samples = args.fid_samples if args.compute_fid else args.ssim_samples
    fid_stats_path = os.path.join(args.run_dir, "fid_stats_cifar10.npz")
    results = {}

    for idx, name in enumerate(methods, 1):
        cfg = METHOD_CONFIGS[name]
        log("\n" + "=" * 80, bold=True)
        log(f"[{idx}/{len(methods)}] Running {name.upper()}", bold=True)
        log(f"Description: {cfg['desc']}")
        log("=" * 80, bold=True)

        # Method output directories (isolated per method so past runs aren't lost)
        prune_dir = os.path.join(args.run_dir, "methods", name, "pruned")
        finetune_dir = os.path.join(args.run_dir, "methods", name, "finetuned")
        sample_dir = os.path.join(args.run_dir, "methods", name, "sample")

        # Step 1: Pruning
        t0 = time.time()
        prune_metrics = run_pruning(name, cfg, args, prune_dir)

        # Step 2: Finetuning
        if not args.skip_finetune:
            run_finetuning(name, prune_dir, finetune_dir, args)
        else:
            log("Skipping finetuning (--skip_finetune is set).")

        # Step 3: Sampling
        run_sampling(name, finetune_dir, prune_dir, sample_dir, n_samples, args)

        # Step 4: SSIM Evaluation
        ssim = evaluate_ssim(sample_dir, baseline_dir)

        # Step 5: Optional FID Evaluation
        fid = float("nan")
        if args.compute_fid:
            fid = evaluate_fid(sample_dir, fid_stats_path, args)

        elapsed = time.time() - t0
        results[name] = {
            "params": prune_metrics["params"],
            "macs": prune_metrics["macs"],
            "jaccard": prune_metrics["jaccard"],
            "spearman": prune_metrics["spearman"],
            "ssim": ssim,
            "fid": fid,
            "elapsed_sec": round(elapsed, 1),
        }

        r = results[name]
        summary_line = (
            f"\n>>> {name}: params={r['params']:.2f}M  macs={r['macs']:.2f}G  "
            f"SSIM={r['ssim']:.4f}  Jaccard={r['jaccard']:.3f}  Spearman={r['spearman']:.3f}"
        )
        if args.compute_fid:
            summary_line += f"  FID={r['fid']:.2f}"
        summary_line += f"  (elapsed: {elapsed/60:.1f}m)"
        log(summary_line, bold=True)

    # Print final summary table
    print_results_table(results, args)

    # Save JSON results
    def clean_for_json(obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        return obj

    json_path = args.save_json or os.path.join(args.run_dir, "results.json")
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(
            clean_for_json({
                "config": {
                    "ratio": args.ratio,
                    "iters": args.iters,
                    "ssim_samples": args.ssim_samples,
                    "compute_fid": args.compute_fid,
                    "device": args.device,
                },
                "results": results,
            }),
            f,
            indent=2,
        )
    log(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()

