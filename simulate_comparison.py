"""
Fixed-timer vs. adaptive signal control in a second-by-second queue
simulation, using the same controllers as the camera demo (controller.py).

Each approach gets random (Poisson) arrivals; while an approach is green it
discharges SATURATION_RATE vehicles per second from its queue. Both
policies run the same two-phase plan (North + South, East + West) and see
exactly the same arrivals - the only difference is which phase gets green
and for how long.

Arrival rates keep the relative share of traffic from the last
detect_and_signal.py run (out.txt), scaled to a realistic, moderately
congested volume - the same scaling sumo_sim/generate_routes.py uses.
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from controller import (  # noqa: E402
    GREEN,
    SATURATION_RATE,
    AdaptiveController,
    FixedTimeController,
    two_phase_plan,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_TXT_PATH = os.path.join(BASE_DIR, "out.txt")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

BASE_VEHS_PER_HOUR = 350   # mean per approach (matches sumo_sim/generate_routes.py)
DEMAND_DURATION = 1800     # seconds of arrivals
MAX_DRAIN = 3600           # extra seconds allowed to clear what's left
DT = 1.0
RANDOM_SEED = 42


def load_baseline_counts():
    if os.path.exists(OUT_TXT_PATH):
        with open(OUT_TXT_PATH, "r") as f:
            try:
                counts = [int(line.strip()) for line in f if line.strip()]
            except ValueError:
                counts = []
        if len(counts) == 4:
            return counts
    # Fallback when detect_and_signal.py hasn't been run yet
    return [23, 41, 12, 26]


def arrival_rates(counts):
    """Vehicles per second per approach, keeping each approach's share."""
    mean = sum(counts) / len(counts) or 1
    return [BASE_VEHS_PER_HOUR / 3600.0 * max(1, c) / mean for c in counts]


def simulate(controller, arrivals):
    """Runs one policy over a fixed arrival schedule (steps x lanes)."""
    num_lanes = arrivals.shape[1]
    queues = [0.0] * num_lanes
    queue_series = []
    vehicle_seconds = 0.0
    total_arrived = 0
    step = 0

    while step < len(arrivals) or (sum(queues) > 0 and step < len(arrivals) + MAX_DRAIN):
        if step < len(arrivals):
            for lane in range(num_lanes):
                queues[lane] += int(arrivals[step, lane])
                total_arrived += int(arrivals[step, lane])

        lights = controller.update(DT, [int(round(q)) for q in queues])
        for lane, light in enumerate(lights):
            if light == GREEN:
                queues[lane] = max(0.0, queues[lane] - SATURATION_RATE * DT)

        vehicle_seconds += sum(queues) * DT
        queue_series.append(round(sum(queues), 2))
        step += 1

    return {
        "queue_over_time": queue_series,
        "avg_queue": float(np.mean(queue_series[:len(arrivals)])),
        "max_queue": float(max(queue_series)),
        "avg_delay": vehicle_seconds / total_arrived if total_arrived else 0.0,
        "clearance_time": step,
        "cleared": bool(sum(queues) <= 0),
        "green_phases": len(controller.phase_log),
    }


def pct_reduction(before, after):
    return (before - after) / before * 100 if before else 0.0


def main():
    counts = load_baseline_counts()
    rates = arrival_rates(counts)
    print("Approach counts (relative demand):", counts)
    print("Arrival rates (veh/h):", [round(r * 3600) for r in rates])

    rng = np.random.default_rng(RANDOM_SEED)
    steps = int(DEMAND_DURATION / DT)
    arrivals = rng.poisson(np.array(rates) * DT, size=(steps, len(rates)))

    # Both policies run the standard two-phase plan (North + South, East + West)
    # with the same yellow / all-red change intervals.
    phases = two_phase_plan(len(rates))
    fixed = simulate(FixedTimeController(len(rates), phases), arrivals)
    adaptive = simulate(AdaptiveController(len(rates), phases), arrivals)

    print("\n--- Results ({} s of arrivals, {} vehicles) ---".format(DEMAND_DURATION, int(arrivals.sum())))
    for label, r in (("Fixed timer", fixed), ("Adaptive", adaptive)):
        print("{:12s} avg queue {:6.1f} | max queue {:6.1f} | avg delay {:6.1f}s | cleared after {} s".format(
            label, r["avg_queue"], r["max_queue"], r["avg_delay"], r["clearance_time"]))
    queue_improvement = pct_reduction(fixed["avg_queue"], adaptive["avg_queue"])
    delay_improvement = pct_reduction(fixed["avg_delay"], adaptive["avg_delay"])
    print("\nAdaptive control: average queue {:.1f}% shorter, average delay {:.1f}% lower.".format(
        queue_improvement, delay_improvement))

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(fixed["queue_over_time"], label="Fixed timer", color="#9aa0a6")
    plt.plot(adaptive["queue_over_time"], label="Adaptive", color="#2e9e4f")
    plt.axvline(DEMAND_DURATION, color="#cccccc", linestyle="--", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Vehicles queued (all approaches)")
    plt.title("Fixed timer vs. adaptive control - total queue")
    plt.legend()
    plt.grid(alpha=0.3)
    chart_path = os.path.join(OUTPUTS_DIR, "comparison_chart.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=120)
    print("\nSaved chart to:", chart_path)

    json_path = os.path.join(OUTPUTS_DIR, "comparison.json")
    with open(json_path, "w") as f:
        json.dump({
            "base_counts": counts,
            "arrival_rates_vph": [round(r * 3600, 1) for r in rates],
            "demand_duration": DEMAND_DURATION,
            "total_vehicles": int(arrivals.sum()),
            "fixed_timer": fixed,
            "adaptive": adaptive,
            "queue_improvement_pct": queue_improvement,
            "delay_improvement_pct": delay_improvement,
        }, f, indent=2)
    print("Saved stats to:", json_path)


if __name__ == "__main__":
    main()
