"""Trigger evaluation harness — offline, no IsaacLab required.

Loads the val split of the HDF5 dataset (same split as training),
runs the visual trigger model, and reports classification metrics +
per-sample inference latency on both CPU and GPU.

Usage (from repo root, in isaaclab or navila conda env):
    python legged-loco/scripts/eval_trigger.py \\
        --h5_path legged-loco/data/obstacle_dataset.h5 \\
        --features_path data/trigger_features_real.npy \\
        --checkpoint checkpoints/trigger_real_visual.pt \\
        --out_dir legged-loco/logs/eval_trigger

Outputs
-------
    legged-loco/logs/eval_trigger/
        RESULTS.md          — metrics summary
        pr_curve.png        — precision-recall curve
        latency.json        — per-sample latency (p50/p95/p99, CPU + GPU)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "trigger_training"))
sys.path.insert(0, os.path.join(
    _REPO_ROOT, "legged-loco", "isaaclab_exts",
    "omni.isaac.leggedloco", "omni", "isaac", "leggedloco",
    "leggedloco", "mdp", "triggers",
))

from data_loader import load_h5_dataset
from train_trigger_visual import split_by_episode, build_arrays
from visual_trigger import VisualTrigger, VisualTriggerCfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5_path",       default="legged-loco/data/obstacle_dataset.h5")
    p.add_argument("--features_path", default="data/trigger_features_real.npy")
    p.add_argument("--checkpoint",    default="checkpoints/trigger_real_visual.pt")
    p.add_argument("--threshold",     type=float, default=0.5)
    p.add_argument("--out_dir",       default="legged-loco/logs/eval_trigger")
    p.add_argument("--latency_n",     type=int,   default=100)
    p.add_argument("--latency_warmup",type=int,   default=10)
    return p.parse_args()


def compute_metrics(y_true, scores, threshold=0.5):
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, precision_recall_curve,
    )
    preds = (scores >= threshold).astype(int)
    prec  = precision_score(y_true, preds, zero_division=0)
    rec   = recall_score(y_true, preds, zero_division=0)
    f1    = f1_score(y_true, preds, zero_division=0)
    try:
        auroc = roc_auc_score(y_true, scores)
    except Exception:
        auroc = float("nan")
    cm    = confusion_matrix(y_true, preds).tolist()
    pr, rc, thr = precision_recall_curve(y_true, scores)
    return dict(precision=prec, recall=rec, f1=f1, auroc=auroc,
                confusion_matrix=cm, pr_curve=(pr.tolist(), rc.tolist(), thr.tolist()))


def benchmark_latency(trigger, features_np, n, warmup, device_str):
    """Benchmark predict_from_features() on a single sample repeated n times."""
    feat = torch.tensor(features_np[0:1], dtype=torch.float32)
    use_gpu = (device_str != "cpu") and torch.cuda.is_available()

    # Warmup
    for _ in range(warmup):
        trigger.predict_from_features(feat, 1.5, 0.1, 0.0, 2.0)
        if use_gpu:
            torch.cuda.synchronize()

    times = []
    for _ in range(n):
        if use_gpu:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        trigger.predict_from_features(feat, 1.5, 0.1, 0.0, 2.0)
        if use_gpu:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    return {
        "device": device_str,
        "mean_ms":  float(np.mean(times)),
        "p50_ms":   float(np.percentile(times, 50)),
        "p95_ms":   float(np.percentile(times, 95)),
        "p99_ms":   float(np.percentile(times, 99)),
        "n":        n,
    }


def benchmark_full_step(trigger, n, warmup):
    """Benchmark full step() including ResNet on a synthetic 320×240 image."""
    rgb = np.random.randint(0, 255, (1, 240, 320, 3), dtype=np.uint8)
    use_gpu = torch.cuda.is_available()

    for _ in range(warmup):
        trigger.step(rgb, np.array([1.5], np.float32),
                     np.array([0.1], np.float32), np.array([2.0], np.float32))
        if use_gpu:
            torch.cuda.synchronize()

    times = []
    for _ in range(n):
        if use_gpu:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        trigger.step(rgb, np.array([1.5], np.float32),
                     np.array([0.1], np.float32), np.array([2.0], np.float32))
        if use_gpu:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "p50_ms":  float(np.percentile(times, 50)),
        "p95_ms":  float(np.percentile(times, 95)),
        "p99_ms":  float(np.percentile(times, 99)),
        "n": n,
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load dataset & split ───────────────────────────────────────────────
    print("[eval_trigger] Loading dataset...")
    dataset, _, _ = load_h5_dataset(args.h5_path, include_images=False)
    features = np.load(args.features_path)   # (N, 512)

    _, val_ids, _ = split_by_episode(dataset, val_frac=0.1, test_frac=0.1, seed=42)
    X_scalar, y, X_visual = build_arrays(dataset, val_ids, features=features)
    print(f"[eval_trigger] Val samples: {len(y)}  "
          f"(positive rate: {y.mean():.3f})")

    # ── Load trigger (GPU) ─────────────────────────────────────────────────
    cfg_gpu = VisualTriggerCfg(checkpoint_path=args.checkpoint,
                               threshold=args.threshold, device="cuda")
    trigger_gpu = VisualTrigger(cfg_gpu, num_envs=1)
    dev = trigger_gpu.device

    # Run inference on val set
    print("[eval_trigger] Running val set inference...")
    X_s_t = torch.tensor(X_scalar, dtype=torch.float32, device=dev)
    X_v_t = torch.tensor(X_visual, dtype=torch.float32, device=dev)
    with torch.no_grad():
        scalars_norm = (X_s_t - trigger_gpu._scalar_mean) / (trigger_gpu._scalar_std + 1e-6)
        x = torch.cat([scalars_norm, X_v_t], dim=1)
        logits = trigger_gpu._mlp(x)
        scores = torch.sigmoid(logits).cpu().numpy()

    metrics = compute_metrics(y.astype(int), scores, threshold=args.threshold)

    # ── Precision-recall curve ─────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pr, rc, _ = metrics["pr_curve"]
        plt.figure(figsize=(6, 5))
        plt.plot(rc, pr, lw=2)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Trigger PR Curve  (AUROC={metrics['auroc']:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "pr_curve.png"), dpi=120)
        plt.close()
        print("[eval_trigger] PR curve saved.")
    except ImportError:
        print("[eval_trigger] matplotlib not available — skipping PR curve plot.")

    # ── Latency benchmarks ─────────────────────────────────────────────────
    print("[eval_trigger] Benchmarking latency (GPU, MLP only)...")
    lat_gpu_mlp = benchmark_latency(trigger_gpu, X_visual,
                                    args.latency_n, args.latency_warmup, "cuda")
    print("[eval_trigger] Benchmarking latency (GPU, full step with ResNet)...")
    lat_gpu_full = benchmark_full_step(trigger_gpu, args.latency_n, args.latency_warmup)

    print("[eval_trigger] Benchmarking latency (CPU)...")
    cfg_cpu = VisualTriggerCfg(checkpoint_path=args.checkpoint,
                               threshold=args.threshold, device="cpu")
    trigger_cpu = VisualTrigger(cfg_cpu, num_envs=1)
    lat_cpu_mlp  = benchmark_latency(trigger_cpu, X_visual,
                                     args.latency_n, args.latency_warmup, "cpu")

    latency = {
        "gpu_mlp_only":  lat_gpu_mlp,
        "gpu_full_step": lat_gpu_full,
        "cpu_mlp_only":  lat_cpu_mlp,
        "note": ("gpu_full_step includes ResNet-18 backbone and is the "
                 "production latency. gpu_mlp_only isolates the MLP head."),
    }
    with open(os.path.join(args.out_dir, "latency.json"), "w") as f:
        json.dump(latency, f, indent=2)

    # ── RESULTS.md ─────────────────────────────────────────────────────────
    cm = metrics["confusion_matrix"]
    tn, fp_n = (cm[0][0], cm[0][1]) if len(cm) > 1 else (0, 0)
    fn, tp   = (cm[1][0], cm[1][1]) if len(cm) > 1 else (0, 0)

    md = f"""# Trigger Evaluation Results

## Dataset
- Source: `{args.h5_path}`
- Split: val (val_frac=0.1, seed=42, episode-based)
- Val samples: {len(y)}  |  Positive rate: {y.mean():.3f}

## Classification Metrics  (threshold={args.threshold})
| Metric | Value |
|---|---|
| Precision | {metrics['precision']:.4f} |
| Recall    | {metrics['recall']:.4f} |
| F1        | {metrics['f1']:.4f} |
| AUROC     | {metrics['auroc']:.4f} |

## Confusion Matrix
|   | Pred 0 | Pred 1 |
|---|---|---|
| True 0 | {tn} | {fp_n} |
| True 1 | {fn} | {tp} |

## Latency (N={args.latency_n}, warmup={args.latency_warmup})
| Mode | p50 [ms] | p95 [ms] | p99 [ms] |
|---|---|---|---|
| GPU — full step (ResNet+MLP) ← production | {lat_gpu_full['p50_ms']:.2f} | {lat_gpu_full['p95_ms']:.2f} | {lat_gpu_full['p99_ms']:.2f} |
| GPU — MLP only | {lat_gpu_mlp['p50_ms']:.2f} | {lat_gpu_mlp['p95_ms']:.2f} | {lat_gpu_mlp['p99_ms']:.2f} |
| CPU — MLP only | {lat_cpu_mlp['p50_ms']:.2f} | {lat_cpu_mlp['p95_ms']:.2f} | {lat_cpu_mlp['p99_ms']:.2f} |

## Completion criterion
- Milestone 2: F1 ≥ 0.8 on val split → **{'PASS' if metrics['f1'] >= 0.8 else 'FAIL'} (F1={metrics['f1']:.4f})**
- Milestone 3 (latency): trigger ≥ 10× faster than NaVILA at p50 → see `compare_latency.py`

PR curve: `pr_curve.png`
Raw latency: `latency.json`
"""
    with open(os.path.join(args.out_dir, "RESULTS.md"), "w") as f:
        f.write(md)

    print(md)
    print(f"[eval_trigger] Results → {args.out_dir}/RESULTS.md")


if __name__ == "__main__":
    main()
