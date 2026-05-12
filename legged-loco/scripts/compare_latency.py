"""Compare trigger vs. NaVILA inference latency.

Reads latency.json from eval_trigger.py and eval_navila.py and produces
a unified comparison table with an empirical conclusion.

Usage:
    python legged-loco/scripts/compare_latency.py \\
        --trigger_json legged-loco/logs/eval_trigger/latency.json \\
        --navila_json  legged-loco/logs/eval_navila/latency.json  \\
        --out          legged-loco/logs/latency_comparison.md
"""

import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trigger_json",
                   default="legged-loco/logs/eval_trigger/latency.json")
    p.add_argument("--navila_json",
                   default="legged-loco/logs/eval_navila/latency.json")
    p.add_argument("--out",
                   default="legged-loco/logs/latency_comparison.md")
    return p.parse_args()


def _fmt(v):
    return f"{v:.2f}" if v is not None else "N/A"


def main():
    args = parse_args()

    with open(args.trigger_json) as f:
        t_data = json.load(f)
    with open(args.navila_json) as f:
        n_data = json.load(f)

    # Trigger: production number is gpu_full_step
    t_gpu = t_data.get("gpu_full_step", {})
    t_cpu = t_data.get("cpu_mlp_only", {})
    # NaVILA: wall-clock socket round-trip on GPU
    n_gpu = n_data.get("gpu_wall_clock", {})

    t_p50 = t_gpu.get("p50_ms")
    t_p95 = t_gpu.get("p95_ms")
    t_p99 = t_gpu.get("p99_ms")
    n_p50 = n_gpu.get("p50_ms")
    n_p95 = n_gpu.get("p95_ms")
    n_p99 = n_gpu.get("p99_ms")

    speedup_p50 = (n_p50 / t_p50) if (t_p50 and n_p50 and t_p50 > 0) else None
    speedup_p95 = (n_p95 / t_p95) if (t_p95 and n_p95 and t_p95 > 0) else None
    speedup_p99 = (n_p99 / t_p99) if (t_p99 and n_p99 and t_p99 > 0) else None

    # Empirical conclusion
    if speedup_p50 is not None:
        if speedup_p50 >= 50:
            verdict = f"a {speedup_p50:.0f}× speedup — strongly justifies the trigger as a real-time gate"
        elif speedup_p50 >= 10:
            verdict = f"a {speedup_p50:.0f}× speedup — justifies the trigger as a practical gate"
        elif speedup_p50 >= 2:
            verdict = (f"only a {speedup_p50:.1f}× speedup — the trigger is faster but "
                       "the margin may not justify the added complexity for all latency budgets")
        else:
            verdict = (f"less than 2× speedup — the trigger does not provide meaningful "
                       "latency advantage over querying NaVILA directly")
    else:
        verdict = "speedup could not be computed (missing data)"

    md = f"""# Trigger vs. NaVILA Latency Comparison

## Raw numbers

| System | Mode | p50 [ms] | p95 [ms] | p99 [ms] |
|---|---|---|---|---|
| Visual Trigger | GPU full step (ResNet+MLP) — **production** | {_fmt(t_p50)} | {_fmt(t_p95)} | {_fmt(t_p99)} |
| Visual Trigger | CPU MLP only | {_fmt(t_cpu.get('p50_ms'))} | {_fmt(t_cpu.get('p95_ms'))} | {_fmt(t_cpu.get('p99_ms'))} |
| NaVILA | GPU wall-clock (socket round-trip) | {_fmt(n_p50)} | {_fmt(n_p95)} | {_fmt(n_p99)} |

## Speedup (NaVILA / Trigger, GPU full-step vs GPU wall-clock)

| Percentile | Speedup |
|---|---|
| p50 | {_fmt(speedup_p50)}× |
| p95 | {_fmt(speedup_p95)}× |
| p99 | {_fmt(speedup_p99)}× |

## Conclusion

At p50, the trigger runs in {_fmt(t_p50)} ms vs. NaVILA at {_fmt(n_p50)} ms —
{verdict}.

The trigger is queried every {t_data.get('trigger_every_steps', 5)} control steps
({t_data.get('trigger_every_steps', 5) * 0.02 * 1000:.0f} ms intervals at 50 Hz),
so its p50 latency of {_fmt(t_p50)} ms fits comfortably within the inter-query budget.
NaVILA is queried much less frequently (on STOP or command expiry), but its per-query
latency dominates the avoidance reaction time — motivating the trigger as a fast,
low-cost early-warning system.

## Source files
- Trigger: `{args.trigger_json}`
- NaVILA:  `{args.navila_json}`
"""

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(md)
    print(md)
    print(f"[compare_latency] → {args.out}")


if __name__ == "__main__":
    main()
