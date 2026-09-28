"""
Before/after comparison: naive fixed-timer signal control vs. the adaptive
saturation-flow algorithm in detect_and_signal.py.

Both schemes face the exact same physics (vehicles arrive, and whatever a
green phase can't clear carries over to the next cycle) - the only thing
that differs is the *policy* deciding how long each lane's green light is.

Arrivals are seeded from the real vehicle counts detected by the last
detect_and_signal.py run (out.txt), with Poisson noise per cycle so the
simulation isn't just repeating one static scenario. Run this after
detect_and_signal.py at least once.
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detect_and_signal import SATURATION_RATE, compute_signal_plan

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_TXT_PATH = os.path.join(BASE_DIR, "out.txt")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

NUM_CYCLES = 25
FIXED_GREEN = 30  # seconds - naive baseline gives every lane the same green every cycle
RANDOM_SEED = 42


def load_baseline_counts():
    if os.path.exists(OUT_TXT_PATH):
        with open(OUT_TXT_PATH, "r") as f:
            counts = [int(line.strip()) for line in f if line.strip()]
        if counts:
            return counts
    # Fall back to a demonstrative scenario if detect_and_signal.py hasn't
    # been run yet (mirrors the "45 vs 25" example from the project brief).
    return [25, 25, 25, 45]


def fixed_timer_plan(vehicle_counts, backlog, state):
    """Naive baseline: every lane always gets the same green time,
    regardless of how much demand it actually has. No aging/escalation -
    that's precisely the adaptive scheme's advantage."""
    plan = []
    new_backlog = []
    for count, waiting in zip(vehicle_counts, backlog):
        demand = count + waiting
        capacity = FIXED_GREEN * SATURATION_RATE
        served = min(demand, capacity)
        leftover = max(0.0, demand - capacity)
        plan.append({"green": FIXED_GREEN, "served": served, "leftover": leftover})
        new_backlog.append(leftover)
    return plan, new_backlog, state


def adaptive_plan(vehicle_counts, backlog, state):
    cycles_waited = state
    plan, new_backlog, new_cycles_waited = compute_signal_plan(vehicle_counts, backlog, cycles_waited)
    return plan, new_backlog, new_cycles_waited


def run_simulation(scheme_fn, base_counts, rng):
    num_lanes = len(base_counts)
    backlog = [0.0] * num_lanes
    state = [0] * num_lanes  # scheme-specific extra state (e.g. cycles_waited)

    total_backlog_per_cycle = []
    total_served_per_cycle = []
    cycle_time_per_cycle = []

    for _ in range(NUM_CYCLES):
        arrivals = [rng.poisson(max(1, c)) for c in base_counts]
        demand_counts = arrivals  # backlog is added inside the scheme itself

        plan, backlog, state = scheme_fn(demand_counts, backlog, state)

        total_backlog_per_cycle.append(sum(backlog))
        total_served_per_cycle.append(sum(p["served"] for p in plan))
        cycle_time_per_cycle.append(sum(p["green"] for p in plan))

    return {
        "backlog_over_time": total_backlog_per_cycle,
        "served_over_time": total_served_per_cycle,
        "cycle_time_over_time": cycle_time_per_cycle,
    }


def summarize(label, result):
    avg_backlog = sum(result["backlog_over_time"]) / len(result["backlog_over_time"])
    final_backlog = result["backlog_over_time"][-1]
    total_served = sum(result["served_over_time"])
    avg_cycle_time = sum(result["cycle_time_over_time"]) / len(result["cycle_time_over_time"])
    print("{}: avg queued vehicles/cycle={:.1f}  final backlog={:.1f}  "
          "total vehicles served={:.1f}  avg cycle time={:.1f}s".format(
              label, avg_backlog, final_backlog, total_served, avg_cycle_time))
    return {
        "avg_backlog": avg_backlog,
        "final_backlog": final_backlog,
        "total_served": total_served,
        "avg_cycle_time": avg_cycle_time,
    }


def main():
    base_counts = load_baseline_counts()
    print("Baseline per-lane vehicle counts (from last real detection run):", base_counts)
    print("Simulating {} signal cycles for each scheme...\n".format(NUM_CYCLES))

    rng_fixed = np.random.default_rng(RANDOM_SEED)
    rng_adaptive = np.random.default_rng(RANDOM_SEED)  # same seed -> same arrivals, fair comparison

    fixed_result = run_simulation(fixed_timer_plan, base_counts, rng_fixed)
    adaptive_result = run_simulation(adaptive_plan, base_counts, rng_adaptive)

    print("--- Results ---")
    fixed_summary = summarize("Fixed-timer (naive, {}s every lane)".format(FIXED_GREEN), fixed_result)
    adaptive_summary = summarize("Adaptive (saturation-flow + carry-over)", adaptive_result)

    backlog_improvement = (
        (fixed_summary["avg_backlog"] - adaptive_summary["avg_backlog"])
        / fixed_summary["avg_backlog"] * 100
        if fixed_summary["avg_backlog"] > 0 else 0.0
    )
    print("\nAverage queue length reduced by {:.1f}% with the adaptive algorithm.".format(backlog_improvement))

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Chart: total backlog (vehicles waiting across all lanes) over cycles
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, NUM_CYCLES + 1), fixed_result["backlog_over_time"], marker="o", label="Fixed-timer (naive)")
    plt.plot(range(1, NUM_CYCLES + 1), adaptive_result["backlog_over_time"], marker="o", label="Adaptive (this project)")
    plt.xlabel("Signal cycle")
    plt.ylabel("Total vehicles waiting across all lanes (backlog)")
    plt.title("Fixed-timer vs. Adaptive Signal Control - Queue Buildup Over Time")
    plt.legend()
    plt.grid(alpha=0.3)
    chart_path = os.path.join(OUTPUTS_DIR, "comparison_chart.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=120)
    print("\nSaved comparison chart to:", chart_path)

    comparison_json_path = os.path.join(OUTPUTS_DIR, "comparison.json")
    with open(comparison_json_path, "w") as f:
        json.dump({
            "base_counts": base_counts,
            "num_cycles": NUM_CYCLES,
            "fixed_timer": {**fixed_summary, **fixed_result},
            "adaptive": {**adaptive_summary, **adaptive_result},
            "backlog_improvement_pct": backlog_improvement,
        }, f, indent=2)
    print("Saved comparison stats to:", comparison_json_path)


if __name__ == "__main__":
    main()
