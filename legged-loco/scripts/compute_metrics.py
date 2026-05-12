"""Compute evaluation metrics from run_eval_baselines output and produce plots.

Discovers all runs under --data_dir (default: logs/eval_baselines/data/),
combines episodes across multiple runs of the same method, then prints a
summary table and saves paper-ready figures.

Usage (from legged-loco/):
    python scripts/compute_metrics.py
    python scripts/compute_metrics.py --data_dir logs/eval_baselines/data --out results/
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

W_T = 1.0    # time weight (matches run_eval_baselines default)
W_E = 0.01   # effort weight
W_D = 0.5    # path-length weight

T_VLA_GPU = 1.5   # seconds per VLA call on GPU
T_VLA_CPU = 8.0   # seconds per VLA call on CPU

# Map directory names → canonical method keys
_DIR_TO_METHOD = {
    "hybrid":    "learned",
    "threshold": "threshold",
    "theshold":  "threshold",   # typo in existing dirs
    "no_vla":    "no_vla",
}

METHOD_ORDER  = ["no_vla", "threshold", "learned"]
METHOD_LABELS = {
    "no_vla":    "No VLA (RRT* oracle)",
    "threshold": "Scalar MLP trigger ",
    "learned":   "Visual trigger — ours",
}
METHOD_COLORS = {
    "no_vla":    "#4CAF50",
    "threshold": "#FF9800",
    "learned":   "#2196F3",
}
METHOD_MARKERS = {
    "no_vla":    "s",
    "threshold": "^",
    "learned":   "o",
}

# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def _load_run(run_dir: Path) -> dict | None:
    """Load one run directory. Returns None if malformed."""
    rj = run_dir / "results.json"
    ep = run_dir / "episodes.csv"
    if not rj.exists() or not ep.exists():
        return None
    with open(rj) as f:
        raw = json.load(f)
    episodes_df = pd.read_csv(ep)

    # Normalize: older runs nest metric lists under raw["metrics"],
    # newer runs put them at the top level.
    results = raw.get("metrics", raw)

    # Spatial CSV is optional (older runs may not have it)
    sp = run_dir / "spatial.csv"
    spatial_df = pd.read_csv(sp) if sp.exists() else None

    return {
        "results":  results,
        "episodes": episodes_df,
        "spatial":  spatial_df,
        "run_dir":  str(run_dir),
    }


def load_all(data_dir: str) -> dict[str, list[dict]]:
    """Return {method: [run_dict, ...]} for every run found under data_dir."""
    data_dir = Path(data_dir)
    runs_by_method: dict[str, list[dict]] = {}

    for method_dir in sorted(data_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        method = _DIR_TO_METHOD.get(method_dir.name)
        if method is None:
            print(f"[WARN] Unknown method dir '{method_dir.name}' — skipping.")
            continue

        for run_dir in sorted(method_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run = _load_run(run_dir)
            if run is None:
                print(f"[WARN] Incomplete run {run_dir} — skipping.")
                continue
            runs_by_method.setdefault(method, []).append(run)
            print(f"  {method:12s}  {run_dir.name}  "
                  f"({len(run['episodes'])} episodes)")

    return runs_by_method


# ──────────────────────────────────────────────────────────────────────
# Metric extraction
# ──────────────────────────────────────────────────────────────────────

def _combine_runs(runs: list[dict]) -> dict:
    """Concatenate all episodes and metric lists from multiple runs."""
    episodes = pd.concat([r["episodes"] for r in runs], ignore_index=True)

    # Collect precomputed per-episode lists from results.json
    rho_J  = []
    spl    = []
    n_vla  = []
    t_wall_sim = []   # simulation time only (steps * dt)
    J_star_vals = []

    for r in runs:
        m = r["results"]
        rho_J.extend(m.get("rho_J_list",  []))
        spl.extend(  m.get("spl_list",    []))
        n_vla.extend(m.get("nvla_list",   []))
        # twall_list in JSON already includes VLA latency at t_bar_vla=1.5s
        # We recompute T_wall from scratch using sim time + N_vla * latency
        # so we can vary t_bar_vla for GPU vs CPU bars.
        J_star_vals.append(m.get("J_star", 32.0))

    J_star = float(np.mean(J_star_vals))

    # Recompute T_wall from episode data so we can use GPU/CPU latencies
    n_vla_arr   = np.array(n_vla, dtype=float)
    time_s_arr  = episodes["time_s"].values.astype(float)
    success_arr = episodes["success"].values.astype(bool)

    # Recompute rho_J directly from CSV so weights are consistent
    path_m  = episodes["path_length_m"].values.astype(float)
    effort  = episodes["effort_sum"].values.astype(float)
    l_fail  = 50.0
    d_goal  = episodes["d_goal_final"].values.astype(float)
    l_term  = np.where(success_arr, 0.0, l_fail + d_goal)
    J_hat   = W_T * time_s_arr + W_D * path_m + W_E * effort + l_term
    rho_J_recomputed = J_hat / J_star

    # SPL: L = arc length of reference trajectory (same J_star basis)
    # Derive L from J_star: J* = W_T*(L/v_nom) + W_D*L  → L = J*/(W_T/v_nom + W_D)
    v_nom = 0.5
    L_ref = J_star / (W_T / v_nom + W_D)
    spl_recomputed = np.where(success_arr, L_ref / np.maximum(path_m, L_ref), 0.0)

    t_wall_gpu = time_s_arr + n_vla_arr * T_VLA_GPU
    t_wall_cpu = time_s_arr + n_vla_arr * T_VLA_CPU

    return {
        "episodes":      episodes,
        "J_star":        J_star,
        "L_ref":         L_ref,
        "rho_J":         rho_J_recomputed,
        "spl":           spl_recomputed,
        "n_vla":         n_vla_arr,
        "t_wall_gpu":    t_wall_gpu,
        "t_wall_cpu":    t_wall_cpu,
        "success":       success_arr,
        "runs":          runs,
    }


# ──────────────────────────────────────────────────────────────────────
# Aggregation → summary DataFrame
# ──────────────────────────────────────────────────────────────────────

def aggregate(runs_by_method: dict) -> tuple[pd.DataFrame, dict]:
    combined = {}
    for method in METHOD_ORDER:
        if method not in runs_by_method:
            continue
        combined[method] = _combine_runs(runs_by_method[method])

    rows = []
    for method, c in combined.items():
        n = len(c["episodes"])
        rows.append({
            "method":          method,
            "label":           METHOD_LABELS[method],
            "n_episodes":      n,
            "SR":              float(c["success"].mean()),
            "SPL_mean":        float(c["spl"].mean()),
            "SPL_std":         float(c["spl"].std()),
            "rho_J_mean":      float(c["rho_J"].mean()),
            "rho_J_std":       float(c["rho_J"].std()),
            "N_vla_mean":      float(c["n_vla"].mean()),
            "N_vla_std":       float(c["n_vla"].std()),
            "T_wall_GPU_mean": float(c["t_wall_gpu"].mean()),
            "T_wall_GPU_std":  float(c["t_wall_gpu"].std()),
            "T_wall_CPU_mean": float(c["t_wall_cpu"].mean()),
            "T_wall_CPU_std":  float(c["t_wall_cpu"].std()),
            "J_star":          c["J_star"],
        })

    return pd.DataFrame(rows), combined


# ──────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────

def plot_scatter(df: pd.DataFrame, out_path: str):
    """ρ_J vs N_vla scatter (main paper figure)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in df.iterrows():
        m = row["method"]
        ax.errorbar(
            row["N_vla_mean"], row["rho_J_mean"],
            xerr=row["N_vla_std"], yerr=row["rho_J_std"],
            fmt=METHOD_MARKERS[m], color=METHOD_COLORS[m],
            markersize=12, capsize=4, capthick=1.5,
            label=row["label"], linewidth=1.5,
        )
    ax.axhline(1.0, color="black", ls="--", alpha=0.4, label="J* (optimal)")
    ax.set_xlabel(r"VLA calls per episode  $N_{vla}$", fontsize=12)
    ax.set_ylabel(r"Cost overhead  $\rho_J = \hat J / J^*$", fontsize=12)
    ax.set_title("Cost overhead vs. VLA invocations\n(lower-left is better)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_wallclock(df: pd.DataFrame, out_path: str):
    """Grouped bar: T_wall GPU vs CPU."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w/2, df["T_wall_GPU_mean"], w, yerr=df["T_wall_GPU_std"],
           capsize=4, label=f"GPU (t̄_vla={T_VLA_GPU}s)", color="#1f77b4", alpha=0.85)
    ax.bar(x + w/2, df["T_wall_CPU_mean"], w, yerr=df["T_wall_CPU_std"],
           capsize=4, label=f"CPU (t̄_vla={T_VLA_CPU}s)", color="#ff7f0e", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(df["label"], rotation=15, ha="right")
    ax.set_ylabel("Mean mission time (s)", fontsize=12)
    ax.set_title("Wall-clock mission time by hardware", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_spl_bar(df: pd.DataFrame, out_path: str):
    """Vertical bar: SPL per method."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(df))
    colors = [METHOD_COLORS[m] for m in df["method"]]
    ax.bar(x, df["SPL_mean"], yerr=df["SPL_std"], capsize=5,
           color=colors, alpha=0.85, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=15, ha="right", fontsize=11)
    ax.set_ylabel("SPL (higher is better)", fontsize=12)
    ax.set_title("Success-weighted Path Length", fontsize=12)
    ax.axhline(1.0, color="black", ls="--", alpha=0.4)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_trajectories(combined: dict, out_path: str):
    """Overlay spatial paths from the latest run per method."""
    methods = [m for m in METHOD_ORDER if m in combined]
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 6), squeeze=False)

    for col, method in enumerate(methods):
        ax = axes[0, col]
        c = combined[method]

        # Use the latest run that has spatial data
        spatial = None
        for run in reversed(c["runs"]):
            if run["spatial"] is not None:
                spatial = run["spatial"]
                break

        if spatial is None:
            ax.set_title(f"{METHOD_LABELS[method]}\n(no spatial data)")
            ax.axis("off")
            continue

        ref = spatial[spatial["type"] == "ref_traj"].sort_values("step")
        ax.plot(ref["x"], ref["y"], "k--", lw=1.5, alpha=0.5, label="Reference P*")
        ax.plot(ref["x"].iloc[0], ref["y"].iloc[0], "ks", ms=8)
        ax.plot(ref["x"].iloc[-1], ref["y"].iloc[-1], "k*", ms=14)

        for ep_idx in spatial["episode"].unique():
            if ep_idx < 0:
                continue
            obs = spatial[(spatial["type"] == "obstacle") & (spatial["episode"] == ep_idx)]
            path = spatial[(spatial["type"] == "robot_path") & (spatial["episode"] == ep_idx)].sort_values("step")
            if not obs.empty:
                ox, oy = float(obs["x"].iloc[0]), float(obs["y"].iloc[0])
                circle = plt.Circle((ox, oy), 0.25, color="#E53935", alpha=0.3)
                ax.add_patch(circle)
                ax.plot(ox, oy, "rx", ms=10, mew=2)
            if not path.empty:
                ax.plot(path["x"].values[::5], path["y"].values[::5],
                        color=METHOD_COLORS[method], lw=1.2, alpha=0.7)

        ep_df  = c["episodes"]
        sr     = c["success"].mean()
        n_ep   = len(ep_df)
        ax.set_title(f"{METHOD_LABELS[method]}\nN={n_ep}  SR={sr:.0%}  "
                     f"SPL={c['spl'].mean():.3f}", fontsize=10)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_per_episode(combined: dict, out_path: str):
    """ρ_J per episode index, all methods on one plot."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, c in combined.items():
        rj = c["rho_J"]
        ax.plot(range(len(rj)), rj, marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method], label=METHOD_LABELS[method],
                lw=1.5, ms=6, alpha=0.85)
    ax.axhline(1.0, color="black", ls="--", alpha=0.4)
    ax.set_xlabel("Episode index", fontsize=12)
    ax.set_ylabel(r"$\rho_J$", fontsize=12)
    ax.set_title("Cost overhead per episode", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_per_episode_twall_nvla(combined: dict, out_path: str):
    """Two-panel: per-episode T_wall (GPU) and N_vla as vertical bar groups."""
    methods = [m for m in METHOD_ORDER if m in combined]
    n_methods = len(methods)
    max_eps = max(len(combined[m]["t_wall_gpu"]) for m in methods)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, max_eps * 0.6), 9),
                                    sharex=False)

    width = 0.8 / n_methods
    ep_counts = {m: len(combined[m]["t_wall_gpu"]) for m in methods}
    max_n = max(ep_counts.values())

    for panel_idx, (ax, key, ylabel, title) in enumerate([
        (ax1, "t_wall_gpu", "T_wall GPU (s)",  "Wall-clock time per episode (GPU, t̄_vla=1.5s)"),
        (ax2, "n_vla",      "N_vla",            "VLA calls per episode"),
    ]):
        for j, method in enumerate(methods):
            c = combined[method]
            vals = c[key]
            n = len(vals)
            x = np.arange(n) + j * width - (n_methods - 1) * width / 2
            bars = ax.bar(x, vals, width=width * 0.9,
                          color=METHOD_COLORS[method], alpha=0.85,
                          label=METHOD_LABELS[method] if panel_idx == 0 else "_")
            # Mark failures
            ep_df = c["episodes"]
            for i, (bar, success) in enumerate(zip(bars, ep_df["success"])):
                if not success:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + ax.get_ylim()[1] * 0.01,
                            "✗", ha="center", va="bottom", fontsize=9,
                            color="red")

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(np.arange(max_n))
        ax.set_xticklabels([str(i) for i in range(max_n)], fontsize=9)
        ax.set_xlabel("Episode index", fontsize=11)

    ax1.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="logs/eval_baselines/data",
                        help="Root containing hybrid/, threshold/, no_vla/ subdirs")
    parser.add_argument("--out", default="results/", help="Output directory for plots/CSV")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    print(f"Scanning {args.data_dir} ...")
    runs_by_method = load_all(args.data_dir)
    if not runs_by_method:
        print("No valid runs found.")
        return

    total_eps = sum(sum(len(r["episodes"]) for r in runs)
                    for runs in runs_by_method.values())
    print(f"\nLoaded {total_eps} episodes across {len(runs_by_method)} methods.\n")

    df, combined = aggregate(runs_by_method)

    # ── Print summary table ────────────────────────────────────────────
    print("=" * 75)
    print(f"{'Method':<30} {'N':>4}  {'SR':>5}  {'SPL':>6}  "
          f"{'ρ_J':>7}  {'N_vla':>6}  {'T_wall(GPU)':>11}  {'T_wall(CPU)':>11}")
    print("-" * 75)
    for _, row in df.iterrows():
        print(f"{row['label']:<30} {row['n_episodes']:>4}  "
              f"{row['SR']:>5.1%}  "
              f"{row['SPL_mean']:>5.3f}±{row['SPL_std']:.3f}  "
              f"{row['rho_J_mean']:>6.2f}±{row['rho_J_std']:.2f}  "
              f"{row['N_vla_mean']:>5.1f}±{row['N_vla_std']:.1f}  "
              f"{row['T_wall_GPU_mean']:>10.1f}  "
              f"{row['T_wall_CPU_mean']:>10.1f}")
    print("=" * 75)

    # ── Save summary CSV ───────────────────────────────────────────────
    csv_path = os.path.join(args.out, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # ── Plots ──────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_scatter(              df,       os.path.join(args.out, "scatter_rhoJ_nvla.png"))
    plot_wallclock(            df,       os.path.join(args.out, "wallclock.png"))
    plot_spl_bar(              df,       os.path.join(args.out, "spl_bar.png"))
    plot_per_episode(          combined, os.path.join(args.out, "per_episode_rhoJ.png"))
    plot_per_episode_twall_nvla(combined, os.path.join(args.out, "per_episode_twall_nvla.png"))
    plot_trajectories(         combined, os.path.join(args.out, "trajectories.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()
