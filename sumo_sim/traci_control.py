"""
Closed-loop SUMO validation of the signal controllers.

Drives a SUMO microscopic traffic simulation through TraCI, one second at a
time: every step it counts the vehicles each "camera" would see (everything
within DETECTION_RANGE_M of the stop line, moving or stopped - the same
thing the camera demo counts inside its road zone), feeds those queues to
the exact same controller the camera demo uses (controller.py), and sets
the traffic light to whatever the controller decides. SUMO then simulates real car-following behavior. An independent,
well-established traffic simulator confirming the result is the strongest
validation available.

Both policies run through the same loop with the same two-phase plan and
change intervals, so the only difference between them is the controller.

Usage:
    python traci_control.py --mode both
    python traci_control.py --mode adaptive --gui
"""

import os
import sys
import argparse
import json
import shutil
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

from controller import GREEN, YELLOW, AdaptiveController, FixedTimeController, two_phase_plan  # noqa: E402

SUMOCFG_PATH = os.path.join(SUMO_SIM_DIR, "intersection.sumocfg")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")

TLS_ID = "center"
# Approach order matches the signal's link indices (incLanes in intersection.net.xml)
LANE_APPROACH_EDGE = ["north_in", "east_in", "south_in", "west_in"]

# What the controller "sees" on each approach. The camera demo counts every
# vehicle inside its road zone, moving or stopped, so SUMO measures the same
# thing: all vehicles within DETECTION_RANGE_M of the stop line (about what
# one camera covers). "halting" counts only stopped vehicles instead.
DETECTION_RANGE_M = 100

STEP = 1.0              # seconds per simulation step (SUMO default step length)
MAX_STEPS = 3 * 3600    # safety cap in case a policy never clears the network
QUEUE_SAMPLE_EVERY = 10 # seconds between queue samples in the saved log


def approach_queues(detection="camera", detection_range=DETECTION_RANGE_M):
    if detection == "halting":
        return [traci.edge.getLastStepHaltingNumber(edge) for edge in LANE_APPROACH_EDGE]
    queues = []
    for edge in LANE_APPROACH_EDGE:
        count = 0
        for lane_index in range(traci.edge.getLaneNumber(edge)):
            lane = "{}_{}".format(edge, lane_index)
            length = traci.lane.getLength(lane)
            for vehicle in traci.lane.getLastStepVehicleIDs(lane):
                if length - traci.vehicle.getLanePosition(vehicle) <= detection_range:
                    count += 1
        queues.append(count)
    return queues


def light_state(lights):
    """Controller lights -> SUMO state string, one character per link."""
    return "".join("G" if light == GREEN else "y" if light == YELLOW else "r" for light in lights)


def run(mode, use_gui=False, detection="camera", detection_range=DETECTION_RANGE_M):
    tripinfo_path = os.path.join(OUTPUTS_DIR, "sumo_tripinfo_{}.xml".format(mode))
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    sumo_binary = _find_sumo_binary("sumo-gui" if use_gui else "sumo")
    traci.start([
        sumo_binary, "-c", SUMOCFG_PATH,
        "--tripinfo-output", tripinfo_path,
        "--no-step-log", "true",
        "--waiting-time-memory", "10000",
        "--end", str(MAX_STEPS),
    ])

    num_lanes = len(LANE_APPROACH_EDGE)
    # North + South and East + West don't conflict (see the junction's
    # foes matrix in intersection.net.xml), so both policies use two phases.
    phases = two_phase_plan(num_lanes)
    ctrl = AdaptiveController(num_lanes, phases) if mode == "adaptive" else FixedTimeController(num_lanes, phases)
    queue_log = []
    steps = 0

    try:
        # Run until every vehicle has completed its trip - stopping early
        # would drop the slowest (most-delayed) vehicles from the averages.
        while traci.simulation.getMinExpectedNumber() > 0 and steps < MAX_STEPS:
            queues = approach_queues(detection, detection_range)
            lights = ctrl.update(STEP, queues)
            # Setting the state directly keeps SUMO's own static program
            # (fixed 30 s phases) from switching the light behind our back.
            traci.trafficlight.setRedYellowGreenState(TLS_ID, light_state(lights))
            traci.simulationStep()
            if steps % QUEUE_SAMPLE_EVERY == 0:
                queue_log.append({"t": steps, "queues": queues})
            steps += 1

        remaining = traci.simulation.getMinExpectedNumber()
        if remaining > 0:
            print("Warning: {} vehicles still in the network after {} s ({} mode).".format(
                remaining, MAX_STEPS, mode))
    finally:
        traci.close()

    stats = _parse_tripinfo(tripinfo_path)
    stats["mode"] = mode
    stats["sim_seconds"] = steps
    stats["green_phases"] = len(ctrl.phase_log)
    stats["phase_log"] = ctrl.phase_log
    stats["queue_log"] = queue_log
    return stats


def _find_sumo_binary(name):
    # SUMO ships "sumo.exe" on Windows and a plain "sumo" elsewhere
    candidates = []
    if SUMO_HOME:
        candidates += [os.path.join(SUMO_HOME, "bin", name + ".exe"),
                       os.path.join(SUMO_HOME, "bin", name)]
    for path in candidates:
        if os.path.isfile(path):
            return path
    found = shutil.which(name)
    if found:
        return found
    sys.exit("Could not find the '{}' binary. Install SUMO (e.g. `pip install eclipse-sumo`) "
             "or set SUMO_HOME.".format(name))


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
