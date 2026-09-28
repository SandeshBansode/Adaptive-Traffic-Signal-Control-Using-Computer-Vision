"""
Closed-loop SUMO validation of the adaptive signal algorithm.

Unlike simulate_comparison.py (which simulates the queueing math directly),
this drives an actual SUMO microscopic traffic simulation through TraCI:
every cycle it reads the REAL simulated queue length on each lane (how many
vehicles are actually halted at the light), feeds that into the exact same
compute_signal_plan() used by detect_and_signal.py, and applies the
resulting green durations to the traffic light - then lets SUMO simulate
real car-following behavior for that duration. This is the strongest
available validation: an independent, well-established traffic simulator
confirms the algorithm actually reduces waiting time for simulated vehicles.

Usage:
    python traci_control.py --mode adaptive
    python traci_control.py --mode fixed
    python traci_control.py --mode adaptive --gui
"""

import os
import sys
import argparse
import json
import xml.etree.ElementTree as ET

try:
    import sumo
    os.environ.setdefault("SUMO_HOME", sumo.SUMO_HOME)
except ImportError:
    pass

SUMO_HOME = os.environ.get("SUMO_HOME")
if SUMO_HOME:
    sys.path.append(os.path.join(SUMO_HOME, "tools"))

import traci  # noqa: E402

SUMO_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SUMO_SIM_DIR)
sys.path.append(PROJECT_DIR)

from detect_and_signal import compute_signal_plan, SATURATION_RATE  # noqa: E402

SUMOCFG_PATH = os.path.join(SUMO_SIM_DIR, "intersection.sumocfg")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")

TLS_ID = "center"
# index of each lane's green phase in intersection.net.xml's tlLogic
LANE_GREEN_PHASE = {0: 0, 1: 2, 2: 4, 3: 6}
YELLOW_PHASE_FOR = {0: 1, 1: 3, 2: 5, 3: 7}
LANE_APPROACH_EDGE = {0: "north_in", 1: "east_in", 2: "south_in", 3: "west_in"}

YELLOW_DURATION = 3
FIXED_GREEN = 30
NUM_CYCLES = 20
MAX_DRAIN_CYCLES = 60  # safety cap in case a policy is genuinely oversaturated and never empties


def run(mode, use_gui=False):
    tripinfo_path = os.path.join(OUTPUTS_DIR, "sumo_tripinfo_{}.xml".format(mode))
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    binary = "sumo-gui" if use_gui else "sumo"
    sumo_binary = os.path.join(SUMO_HOME, "bin", binary + ".exe")

    traci.start([
        sumo_binary, "-c", SUMOCFG_PATH,
        "--tripinfo-output", tripinfo_path,
        "--no-step-log", "true",
        "--waiting-time-memory", "1000",
    ])

    backlog = [0.0, 0.0, 0.0, 0.0]
    cycles_waited = [0, 0, 0, 0]
    cycle_log = []

    def run_one_cycle(cycle_idx):
        nonlocal backlog, cycles_waited

        queue_counts = [
            traci.edge.getLastStepHaltingNumber(LANE_APPROACH_EDGE[i]) for i in range(4)
        ]

        if mode == "adaptive":
            plan, backlog, cycles_waited = compute_signal_plan(queue_counts, backlog, cycles_waited)
            greens = [p["green"] for p in plan]
        else:
            # naive fixed-timer baseline: same round-robin phase order,
            # but every lane always gets the same green duration
            greens = [FIXED_GREEN] * 4
            backlog = [max(0.0, q - g * SATURATION_RATE) for q, g in zip(queue_counts, greens)]

        cycle_log.append({"cycle": cycle_idx, "queue_counts": queue_counts, "greens": greens})

        for lane_idx in range(4):
            traci.trafficlight.setPhase(TLS_ID, LANE_GREEN_PHASE[lane_idx])
            _advance(int(round(greens[lane_idx])))
            traci.trafficlight.setPhase(TLS_ID, YELLOW_PHASE_FOR[lane_idx])
            _advance(YELLOW_DURATION)

    try:
        for cycle in range(NUM_CYCLES):
            run_one_cycle(cycle)

        # Keep running full signal cycles - under the SAME algorithm being
        # tested - until (almost) every vehicle has actually completed its
        # trip. Without this, any vehicle still queued when we stop never
        # gets a tripinfo entry at all, so "average waiting time" would
        # silently only reflect the lucky vehicles that finished in time -
        # exactly backwards, since a worse policy leaves MORE stragglers.
        drain_cycle = NUM_CYCLES
        while traci.simulation.getMinExpectedNumber() > 0 and drain_cycle < NUM_CYCLES + MAX_DRAIN_CYCLES:
            run_one_cycle(drain_cycle)
            drain_cycle += 1

        remaining = traci.simulation.getMinExpectedNumber()
        if remaining > 0:
            print("Warning: {} vehicles still in the network after {} drain cycles ({} mode) - "
                  "network is oversaturated for this policy.".format(remaining, MAX_DRAIN_CYCLES, mode))
    finally:
        traci.close()

    stats = _parse_tripinfo(tripinfo_path)
    stats["mode"] = mode
    stats["cycle_log"] = cycle_log
    return stats


def _advance(seconds):
    for _ in range(seconds):
        if traci.simulation.getMinExpectedNumber() <= 0:
            break
        traci.simulationStep()


def _parse_tripinfo(path):
    if not os.path.exists(path):
        return {"completed_trips": 0, "avg_waiting_time": 0.0, "avg_duration": 0.0, "avg_time_loss": 0.0}

    tree = ET.parse(path)
    trips = tree.getroot().findall("tripinfo")

    if not trips:
        return {"completed_trips": 0, "avg_waiting_time": 0.0, "avg_duration": 0.0, "avg_time_loss": 0.0}

    waiting_times = [float(t.get("waitingTime")) for t in trips]
    durations = [float(t.get("duration")) for t in trips]
    time_losses = [float(t.get("timeLoss")) for t in trips]

    return {
        "completed_trips": len(trips),
        "avg_waiting_time": sum(waiting_times) / len(trips),
        "avg_duration": sum(durations) / len(trips),
        "avg_time_loss": sum(time_losses) / len(trips),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fixed", "adaptive", "both"], default="both")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    modes = ["fixed", "adaptive"] if args.mode == "both" else [args.mode]
    results = {}

    for mode in modes:
        print("\n--- Running SUMO with {} signal policy ---".format(mode))
        stats = run(mode, use_gui=args.gui)
        results[mode] = stats
        print("{}: completed_trips={} avg_waiting_time={:.1f}s avg_duration={:.1f}s avg_time_loss={:.1f}s".format(
            mode, stats["completed_trips"], stats["avg_waiting_time"], stats["avg_duration"], stats["avg_time_loss"]))

    if "fixed" in results and "adaptive" in results:
        improvement = (
            (results["fixed"]["avg_waiting_time"] - results["adaptive"]["avg_waiting_time"])
            / results["fixed"]["avg_waiting_time"] * 100
            if results["fixed"]["avg_waiting_time"] > 0 else 0.0
        )
        print("\nSUMO-simulated average waiting time reduced by {:.1f}% with the adaptive algorithm.".format(improvement))
        results["waiting_time_improvement_pct"] = improvement

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(os.path.join(OUTPUTS_DIR, "sumo_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved SUMO comparison to outputs/sumo_comparison.json")


if __name__ == "__main__":
    main()
