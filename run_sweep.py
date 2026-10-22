#!/usr/bin/env python
"""
Unified Sweep Benchmark for Diffusion Model Pruning & Finetuning
================================================================

Orchestrates multi-method, multi-ratio, and multi-step evaluation:
- Evaluates across pruning ratios (e.g. 0.1, 0.2, 0.3)
- Evaluates across finetuning steps (e.g. 0, 1000, 2500, 5000)
- "Train Once, Evaluate Many": Trains once up to max(steps) and saves
  milestone checkpoints, avoiding redundant training.
- Reuses cached baseline samples & precomputed FID statistics.
- Exports results to CSV and JSON with full resume support.

Usage:
  # Quick sweep (step 0 and 1000 steps, ratios 0.1 and 0.3):
  python run_sweep.py --methods magnitude diffpruning --ratios 0.1 0.3 --steps 0 1000

  # Full sweep with FID and resume:
  python run_sweep.py --methods all --ratios 0.1 0.2 0.3 --steps 0 1000 2500 5000 --compute_fid --resume
"""

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time

from run_workflow import (
    ALL_METHODS,
    METHOD_CONFIGS,
    count_images_in_dir,
    ensure_baseline_samples,
    ensure_setup,
    evaluate_fid,
    evaluate_ssim,
    load_method_meta,
    log,
    run_command,
    run_finetuning,
    run_pruning,
    run_sampling,
    save_method_meta,
)


def find_step_checkpoint(finetune_dir, prune_dir, step, use_ema=False):
    """Find checkpoint corresponding to a specific finetune step."""
    if step == 0:
        raw_ckpt = os.path.join(prune_dir, "pruned", "unet_pruned.pth")
        if os.path.exists(raw_ckpt):
            return raw_ckpt, prune_dir
        raise FileNotFoundError(f"Raw pruned checkpoint not found at {raw_ckpt}")

    pruned_dir = os.path.join(finetune_dir, "pruned")
    prefix = "unet_ema_pruned" if use_ema else "unet_pruned"

    # 1. Exact step checkpoint: e.g. unet_pruned-1000.pth
    step_ckpt = os.path.join(pruned_dir, f"{prefix}-{step}.pth")
    if os.path.exists(step_ckpt):
        return step_ckpt, finetune_dir

    # 2. Final checkpoint fallback (if step is max iters)
    final_ckpt = os.path.join(pruned_dir, f"{prefix}.pth")
    if os.path.exists(final_ckpt):
        return final_ckpt, finetune_dir

    raise FileNotFoundError(
        f"Could not find checkpoint for step {step} (checked {step_ckpt} and {final_ckpt})"
    )


def calculate_save_steps(steps, max_step=0, override=None):
    """Calculate checkpoint save frequency to align with requested steps."""
    if override is not None:
        return override
    pos_steps = [s for s in steps if s > 0]
    if not pos_steps:
        return 1000
    gcd_val = pos_steps[0]
    for s in pos_steps[1:]:
        gcd_val = math.gcd(gcd_val, s)
    if max_step > 0 and max_step < 100:
        return max(1, gcd_val)
    return max(100, gcd_val)


def save_sweep_results(results_list, output_dir, compute_fid=False):
    """Save results as both JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "sweep_summary.json")
    csv_path = os.path.join(output_dir, "sweep_summary.csv")

    def clean_val(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    clean_results = [
        {k: clean_val(v) for k, v in row.items()} for row in results_list
    ]
    with open(json_path, "w") as f:
        json.dump(clean_results, f, indent=2)

    # Write CSV
    cols = ["method", "ratio", "step", "params", "macs", "ssim"]
    if compute_fid:
        cols.append("fid")
    cols.extend(["jaccard", "spearman", "elapsed_sec"])

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in results_list:
            writer.writerow(row)


def print_sweep_table(results_list, compute_fid=False):
    """Print an aligned ASCII table grouped by Ratio and Step."""
    if not results_list:
        return

    cols = ["ratio", "step", "method", "params", "macs", "ssim"]
    if compute_fid:
        cols.append("fid")

    sep = "=" * 80
    log("\n" + sep, bold=True)
    log(" SWEEP BENCHMARK RESULTS ", bold=True)
    log(sep)

    header = (
        f"{'ratio':>7s} {'step':>6s} {'method':<16s} {'params(M)':>10s} "
        f"{'macs(G)':>9s} {'ssim':>9s}"
    )
    if compute_fid:
        header += f" {'fid':>9s}"
    log(header, bold=True)
    log("-" * 80)

    # Sort results by ratio, then step, then method
    sorted_res = sorted(
        results_list,
        key=lambda r: (r.get("ratio", 0), r.get("step", 0), r.get("method", ""))
    )

    prev_group = None
    for r in sorted_res:
        group = (r.get("ratio"), r.get("step"))
        if prev_group is not None and group != prev_group:
            print("-" * 80)
        prev_group = group

        row_str = (
            f"{r['ratio']:7.2f} {r['step']:6d} {r['method']:<16s} "
            f"{r['params']:10.2f} {r['macs']:9.2f} {r['ssim']:9.4f}"
        )
        if compute_fid:
            fid_val = r.get("fid", float("nan"))
            row_str += (
                f"{fid_val:9.2f}" if not math.isnan(fid_val) else f"{'nan':>9s}"
            )
        print(row_str)

    log(sep + "\n", bold=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Multi-Ratio & Multi-Step Sweep for Diffusion Pruning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help=f"Methods to evaluate. Choose 'all' or subset of: {ALL_METHODS}",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3],
        help="Pruning ratios to sweep over (e.g. 0.1 0.2 0.3)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[0, 1000],
        help="Finetune steps to evaluate (e.g. 0 1000 2500 5000; 0 = raw pruned)",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Checkpoint save frequency (default: automatically computed from steps)",
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Sampling & FID batch size")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Finetuning batch size")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers for finetuning")
    parser.add_argument("--ddim_steps", type=int, default=100, help="DDIM sampling steps")
    parser.add_argument("--ssim_samples", type=int, default=1000, help="Paired sample count for SSIM")
    parser.add_argument("--compute_fid", action="store_true", default=False, help="Compute FID score")
    parser.add_argument("--fid_samples", type=int, default=10000, help="Sample count when computing FID")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if subprocess.run("which nvidia-smi", shell=True, capture_output=True).returncode == 0 else "cpu",
        help="Device to use (e.g. cuda:0, cpu)",
    )
    parser.add_argument("--dataset", type=str, default="data/cifar10_images", help="Dataset directory")
    parser.add_argument("--model_path", type=str, default="pretrained/ddpm_ema_cifar10", help="Pretrained model dir")
    parser.add_argument("--output_dir", type=str, default="sweep_results", help="Base directory for sweep outputs")
    parser.add_argument("--setup", action="store_true", help="Download CIFAR10 and convert pretrained model if missing")
    parser.add_argument("--force_resample_baseline", action="store_true", help="Force re-sampling pretrained baseline")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume previous runs by skipping stages whose outputs match")
    parser.add_argument("--force_resample", action="store_true", default=False, help="Force re-sampling method images even when resuming")
    parser.add_argument("--use_ema", action="store_true", default=False, help="Enable EMA during finetuning")
    parser.add_argument("--thr", type=float, default=0.05, help="Threshold for diff-pruning stopping criterion")

    # MI hyperparams
    parser.add_argument("--mi_num_batches", type=int, default=32, help="Calibration batches for MI")
    parser.add_argument("--mi_num_locations", type=int, default=4, help="Spatial locations sampled per image for MI")

    return parser.parse_args()


def main():
    args = parse_args()
    args.run_dir = args.output_dir

    # Determine methods
    if "all" in args.methods:
        methods = ALL_METHODS
    else:
        methods = [m for m in args.methods if m in METHOD_CONFIGS]
        if not methods:
            log(f"No valid methods specified in {args.methods}. Valid: {ALL_METHODS}", bold=True)
            sys.exit(1)

    steps = sorted(list(set(args.steps)))
    pos_steps = [s for s in steps if s > 0]
    max_step = max(pos_steps) if pos_steps else 0
    save_steps = calculate_save_steps(steps, max_step=max_step, override=args.save_steps)

    log("=" * 80, bold=True)
    log(" Diffusion Pruning Matrix Sweep ", bold=True)
    log("=" * 80, bold=True)
    log(f"Methods to sweep : {methods}")
    log(f"Pruning ratios   : {args.ratios}")
    log(f"Evaluation steps : {steps} (max training iters: {max_step}, checkpoint freq: {save_steps})")
    log(f"SSIM samples     : {args.ssim_samples}")
    log(f"Compute FID      : {args.compute_fid} (samples: {args.fid_samples if args.compute_fid else 'N/A'})")
    log(f"Output directory : {args.output_dir}")
    log(f"Resume enabled   : {args.resume}")
    log(f"Device           : {args.device}")

    # Step 0: Ensure dataset & model are ready
    ensure_setup(args)

    # Step 1: Ensure baseline samples are ready (shared globally across the sweep)
    baseline_args = copy.deepcopy(args)
    baseline_args.run_dir = args.output_dir
    baseline_dir = ensure_baseline_samples(baseline_args)

    # Step 2: Ensure FID ground truth stats are ready if FID requested
    fid_stats_path = os.path.join(args.output_dir, "fid_stats_cifar10.npz")
    if args.compute_fid:
        if not os.path.exists(fid_stats_path):
            log(f"\nFID stats not found at {fid_stats_path}. Precomputing stats...")
            os.makedirs(args.output_dir, exist_ok=True)
            run_command(
                f"python fid_score.py --save-stats {args.dataset} {fid_stats_path} "
                f"--device {args.device} --batch-size {args.batch_size}",
                desc="Precomputing CIFAR-10 FID statistics",
            )
        else:
            log(f"  [OK] FID stats found at {fid_stats_path}")

    n_samples = args.fid_samples if args.compute_fid else args.ssim_samples
    all_results = []

    # Outer Loop: Ratios
    for r_idx, ratio in enumerate(args.ratios, 1):
        log("\n" + "#" * 80, bold=True)
        log(f" [RATIO {r_idx}/{len(args.ratios)}] Sweep ratio = {ratio}", bold=True)
        log("#" * 80, bold=True)

        ratio_dir = os.path.join(args.output_dir, f"ratio_{ratio}")

        # Middle Loop: Methods
        for m_idx, name in enumerate(methods, 1):
            cfg = METHOD_CONFIGS[name]
            log("\n" + "=" * 80, bold=True)
            log(f" [{m_idx}/{len(methods)}] Ratio {ratio} | Method: {name.upper()}", bold=True)
            log(f" Description: {cfg['desc']}")
            log("=" * 80, bold=True)

            method_dir = os.path.join(ratio_dir, "methods", name)
            prune_dir = os.path.join(method_dir, "pruned")
            finetune_dir = os.path.join(method_dir, "finetuned")
            meta = load_method_meta(method_dir)

            # --- Stage A: Pruning (Once per ratio + method) ---
            pruned_ckpt = os.path.join(prune_dir, "pruned", "unet_pruned.pth")
            prune_matches = (
                meta.get("ratio") == ratio
                and meta.get("model_path") == args.model_path
                and "prune_metrics" in meta
            )
            if cfg["pruner"] == "diff-pruning":
                prune_matches = prune_matches and (meta.get("thr") == args.thr)

            if args.resume and os.path.exists(pruned_ckpt) and prune_matches:
                log(f"  [Resume] Reusing pruned model & metrics (ratio={ratio}). Skipping pruning.")
                prune_metrics = meta["prune_metrics"]
            else:
                ratio_args = copy.deepcopy(args)
                ratio_args.ratio = ratio
                prune_metrics = run_pruning(name, cfg, ratio_args, prune_dir)
                meta["ratio"] = ratio
                meta["model_path"] = args.model_path
                if cfg["pruner"] == "diff-pruning":
                    meta["thr"] = args.thr
                meta["prune_metrics"] = prune_metrics
                save_method_meta(method_dir, meta)

            # --- Stage B: Finetuning (Train Once to max_step) ---
            if max_step > 0:
                expected_ckpt = os.path.join(
                    finetune_dir, "pruned",
                    "unet_ema_pruned.pth" if args.use_ema else "unet_pruned.pth"
                )
                finetune_matches = (
                    meta.get("iters") == max_step
                    and meta.get("use_ema") == args.use_ema
                )
                if args.resume and os.path.exists(expected_ckpt) and finetune_matches:
                    log(f"  [Resume] Reusing finetuned model ({max_step} iters). Skipping training.")
                else:
                    train_args = copy.deepcopy(args)
                    train_args.iters = max_step
                    train_args.save_model_steps = save_steps
                    run_finetuning(name, prune_dir, finetune_dir, train_args)
                    meta["iters"] = max_step
                    meta["use_ema"] = args.use_ema
                    save_method_meta(method_dir, meta)

            # --- Stage C: Inner Loop - Evaluate Each Step Milestone ---
            for step in steps:
                eval_dir = os.path.join(method_dir, f"eval_step_{step}")
                sample_dir = os.path.join(eval_dir, "sample")
                eval_meta = load_method_meta(eval_dir)

                log(f"\n  >> Evaluating {name.upper()} @ Ratio={ratio}, Step={step}...", bold=True)

                t0 = time.time()
                try:
                    ckpt, model_path = find_step_checkpoint(
                        finetune_dir, prune_dir, step, use_ema=args.use_ema
                    )
                except FileNotFoundError as e:
                    log(f"  [Warning] Skipping step {step} for {name}: {e}")
                    continue

                # Sampling
                existing_samples = count_images_in_dir(sample_dir)
                sample_matches = (
                    eval_meta.get("ddim_steps") == args.ddim_steps
                    and existing_samples >= n_samples
                )
                if args.resume and not args.force_resample and sample_matches:
                    log(f"  [Resume] Reusing {existing_samples} existing samples for step {step}. Skipping sampling.")
                else:
                    sample_args = copy.deepcopy(args)
                    run_sampling(
                        name, finetune_dir, prune_dir, sample_dir, n_samples, sample_args,
                        ckpt=ckpt, model_path=model_path,
                    )
                    eval_meta["ddim_steps"] = args.ddim_steps
                    eval_meta["sample_count"] = count_images_in_dir(sample_dir)
                    save_method_meta(eval_dir, eval_meta)

                # SSIM Evaluation
                if args.resume and eval_meta.get("ssim") is not None and not math.isnan(eval_meta.get("ssim", float("nan"))):
                    ssim = eval_meta["ssim"]
                    log(f"  [Resume] Reusing cached SSIM: {ssim:.4f}")
                else:
                    ssim = evaluate_ssim(sample_dir, baseline_dir)
                    eval_meta["ssim"] = ssim
                    save_method_meta(eval_dir, eval_meta)

                # FID Evaluation
                fid = float("nan")
                if args.compute_fid:
                    if args.resume and eval_meta.get("fid") is not None and not math.isnan(eval_meta.get("fid", float("nan"))):
                        fid = eval_meta["fid"]
                        log(f"  [Resume] Reusing cached FID: {fid:.2f}")
                    else:
                        fid = evaluate_fid(sample_dir, fid_stats_path, args)
                        eval_meta["fid"] = fid
                        save_method_meta(eval_dir, eval_meta)

                elapsed = round(time.time() - t0, 1)

                point_result = {
                    "method": name,
                    "ratio": ratio,
                    "step": step,
                    "params": prune_metrics["params"],
                    "macs": prune_metrics["macs"],
                    "jaccard": prune_metrics.get("jaccard", float("nan")),
                    "spearman": prune_metrics.get("spearman", float("nan")),
                    "ssim": ssim,
                    "fid": fid,
                    "elapsed_sec": elapsed,
                }
                all_results.append(point_result)

                # Update live results CSV & JSON after each evaluation
                save_sweep_results(all_results, args.output_dir, compute_fid=args.compute_fid)

                summary_str = (
                    f"     => {name} (r={ratio}, step={step}): "
                    f"params={point_result['params']:.2f}M  macs={point_result['macs']:.2f}G  "
                    f"SSIM={point_result['ssim']:.4f}"
                )
                if args.compute_fid:
                    summary_str += f"  FID={point_result['fid']:.2f}"
                log(summary_str, bold=True)

    # Print consolidated summary table
    print_sweep_table(all_results, compute_fid=args.compute_fid)
    save_sweep_results(all_results, args.output_dir, compute_fid=args.compute_fid)
    log(f"Sweep completed! Results saved to:\n  - {os.path.join(args.output_dir, 'sweep_summary.csv')}\n  - {os.path.join(args.output_dir, 'sweep_summary.json')}\n", bold=True)


if __name__ == "__main__":
    main()
