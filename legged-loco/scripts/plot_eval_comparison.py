"""Plot all three baseline methods on a single spatial comparison figure.

Reads the latest spatial.csv from logs/eval_baselines/{method}/ for each method
and overlays: reference trajectory, robot paths, and obstacle positions.

Usage (from legged-loco/):
    python scripts/plot_eval_comparison.py
    python scripts/plot_eval_comparison.py --out comparison.png
    python scripts/plot_eval_comparison.py --runs hybrid=20260503_182114 threshold=20260503_182514 no_vla=20260503_183342
"""

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ["hybrid", "threshold", "no_vla"]
METHOD_LABELS = {
    "hybrid":    "Proposed (visual trigger + NaVILA)",
    "threshold": "Baseline 2 (scalar MLP trigger + NaVILA)",
    "no_vla":    "Baseline 4 (RRT* oracle, no VLA)",
}
METHOD_COLORS = {
    "hybrid":    "#2196F3",   # blue
    "threshold": "#FF9800",   # orange
    "no_vla":    "#4CAF50",   # green
}
OBSTACLE_COLOR = "#E53935"
REF_COLOR      = "#9E9E9E"
OBSTACLE_RADIUS = 0.25        # box half-width for patch (visual only)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_run(log_root: str, method: str) -> str | None:
    d = os.path.join(log_root, method)
    if not os.path.isdir(d):
        return None
    runs = sorted(r for r in os.listdir(d) if os.path.isdir(os.path.join(d, r)))
    for run in reversed(runs):
        if os.path.exists(os.path.join(d, run, "spatial.csv")):
            return os.path.join(d, run)
    return None


def _load(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(run_dir, "spatial.csv"))


def _load_results(run_dir: str) -> dict:
    import json
    p = os.path.join(run_dir, "results.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot baseline comparison.")
    parser.add_argument("--log_root", default=None,
                        help="Path to logs/eval_baselines/ (auto-detected if omitted)")
    parser.add_argument("--runs", nargs="*", default=None,
                        metavar="METHOD=TIMESTAMP",
                        help="Pin specific run dirs, e.g. hybrid=20260503_182114")
    parser.add_argument("--out", default="eval_comparison.png",
                        help="Output image path")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--downsample", type=int, default=5,
                        help="Plot every Nth robot_path point (reduces clutter)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_root = args.log_root or os.path.join(script_dir, "..", "logs", "eval_baselines")
    log_root = os.path.normpath(log_root)

    # Resolve run directories
    pinned = {}
    if args.runs:
        for spec in args.runs:
            m, ts = spec.split("=", 1)
            pinned[m] = os.path.join(log_root, m, ts)

    run_dirs = {}
    for method in METHODS:
        if method in pinned:
            run_dirs[method] = pinned[method]
        else:
            d = _latest_run(log_root, method)
            if d:
                run_dirs[method] = d
            else:
                print(f"[WARN] No spatial.csv found for method '{method}' — skipping.")

    if not run_dirs:
        print("No runs found. Run the eval script first.")
        sys.exit(1)

    # ── Load data ────────────────────────────────────────────────────────────
    dfs = {m: _load(d) for m, d in run_dirs.items()}
    results = {m: _load_results(d) for m, d in run_dirs.items()}

    # Reference trajectory is the same across methods — take from any
    first_df = next(iter(dfs.values()))
    ref = first_df[first_df["type"] == "ref_traj"].sort_values("step")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))

    # Reference trajectory
    ax.plot(ref["x"], ref["y"],
            color=REF_COLOR, lw=1.5, ls="--", zorder=1, label="Reference P*")
    ax.plot(ref["x"].iloc[0],  ref["y"].iloc[0],
            "s", color=REF_COLOR, ms=8, zorder=5)
    ax.plot(ref["x"].iloc[-1], ref["y"].iloc[-1],
            "*", color=REF_COLOR, ms=14, zorder=5)

    # Per-method paths
    for method, df in dfs.items():
        color = METHOD_COLORS[method]

        # Obstacle (episode 0)
        obs = df[(df["type"] == "obstacle") & (df["episode"] == 0)]
        if not obs.empty:
            ox, oy = float(obs["x"].iloc[0]), float(obs["y"].iloc[0])
            rect = mpatches.Rectangle(
                (ox - OBSTACLE_RADIUS, oy - OBSTACLE_RADIUS),
                2 * OBSTACLE_RADIUS, 2 * OBSTACLE_RADIUS,
                linewidth=1.5, edgecolor=OBSTACLE_COLOR,
                facecolor=OBSTACLE_COLOR, alpha=0.25, zorder=3,
            )
            ax.add_patch(rect)
            ax.plot(ox, oy, "X", color=OBSTACLE_COLOR, ms=10, zorder=6)

        # Robot path
        path = df[(df["type"] == "robot_path") & (df["episode"] == 0)].sort_values("step")
        xs = path["x"].values[::args.downsample]
        ys = path["y"].values[::args.downsample]

        # Build label with key metrics
        label = METHOD_LABELS[method]
        r = results.get(method, {})
        eps = r.get("episodes", [{}])
        ep0 = eps[0] if eps else {}
        ok  = "✓" if ep0.get("success") else "✗"
        nvla = ep0.get("n_vla", "?")
        path_m = ep0.get("path_length_m", None)
        if path_m is not None:
            label += f"\n  {ok}  path={path_m:.2f}m  N_vla={nvla}"

        ax.plot(xs, ys, color=color, lw=1.8, alpha=0.85, zorder=4, label=label)

        # Direction arrows every ~50 points
        skip = max(1, len(xs) // 8)
        for i in range(0, len(xs) - 1, skip):
            dx = xs[min(i + 1, len(xs) - 1)] - xs[i]
            dy = ys[min(i + 1, len(ys) - 1)] - ys[i]
            norm = np.hypot(dx, dy)
            if norm > 1e-4:
                ax.annotate("", xy=(xs[i] + dx / norm * 0.15, ys[i] + dy / norm * 0.15),
                            xytext=(xs[i], ys[i]),
                            arrowprops=dict(arrowstyle="-|>", color=color,
                                            lw=1.2, mutation_scale=10),
                            zorder=5)

    # Obstacle legend patch
    obs_patch = mpatches.Patch(facecolor=OBSTACLE_COLOR, alpha=0.4,
                               edgecolor=OBSTACLE_COLOR, label="Obstacle")

    # ── Formatting ───────────────────────────────────────────────────────────
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles + [obs_patch], labels_leg + ["Obstacle"],
              loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_xlabel("x [m]", fontsize=11)
    ax.set_ylabel("y [m]", fontsize=11)
    ax.set_title("Baseline Comparison — Robot Paths (episode 0)", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"[INFO] Saved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
