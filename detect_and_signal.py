"""
Adaptive Traffic Signal Control - Vehicle Detection UI

Runs YOLOv4-tiny (via OpenCV DNN) over each lane's video, draws bounding
boxes around detected vehicles, tracks/counts them with the project's
own centroid tracker, then computes and displays adaptive green-signal
timings for all lanes.

Replaces the old tensorflow==2.1.0 + dlib + tensornets pipeline
(multithreading.py), which cannot install on Python 3.12/3.14.
"""

import os
import time
import json
import csv
import cv2
import numpy as np

from tracking.centroidtracker import CentroidTracker
from tracking.trackableobject import TrackableObject
from emergency import EmergencyDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")

CFG_PATH = os.path.join(MODELS_DIR, "yolov4-tiny.cfg")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "yolov4-tiny.weights")
NAMES_PATH = os.path.join(MODELS_DIR, "coco.names")

CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4
INPUT_SIZE = 416
DETECT_EVERY_N_FRAMES = 2

VEHICLE_CLASSES = {"bicycle", "car", "motorbike", "bus", "truck"}
BOX_COLOR = (0, 255, 0)
TRACK_COLOR = (255, 0, 0)
TEXT_COLOR = (0, 0, 255)


def load_class_names():
    with open(NAMES_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_network():
    net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layer_names = net.getLayerNames()
    out_layers = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in out_layers.flatten()]
    return net, output_layers


def detect_vehicles(net, output_layers, class_names, frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            class_name = class_names[class_id]

            if confidence >= CONFIDENCE_THRESHOLD and class_name in VEHICLE_CLASSES:
                cx, cy, w, h = detection[0:4] * np.array([width, height, width, height])
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(confidence)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    rects = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            startX, startY, endX, endY = x, y, x + w, y + h
            rects.append((startX, startY, endX, endY))

    return rects


TILE_WIDTH = 640
TILE_HEIGHT = 360
GRID_WINDOW_NAME = "Adaptive Traffic Signal - Live Detection (4 Lanes)"


class LaneState:
    def __init__(self, video_path, label):
        self.video_path = video_path
        self.label = label
        self.cap = cv2.VideoCapture(video_path) if os.path.exists(video_path) else None
        self.ct = CentroidTracker(maxDisappeared=15, maxDistance=75)
        self.trackable_objects = {}
        self.total = 0
        self.frame_count = 0
        self.last_rects = []
        self.finished = self.cap is None
        self.last_tile = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap is not None else 30.0
        if not self.fps or self.fps <= 1:
            self.fps = 30.0
        self.emergency = EmergencyDetector()
        self.emergency_ever_detected = False


def render_lane_tile(state, net, output_layers, class_names):
    if state.finished:
        return draw_overlay(state.last_tile.copy(), state)

    ret, frame = state.cap.read()
    if not ret:
        state.finished = True
        state.cap.release()
        return draw_overlay(state.last_tile.copy(), state, finished=True)

    state.frame_count += 1
    orig_h, orig_w = frame.shape[:2]

    # Detect on the full-resolution frame (small/distant vehicles keep more
    # detail than if we downscaled to the tile size first), then scale the
    # resulting boxes into tile coordinates for drawing/display.
    if state.frame_count % DETECT_EVERY_N_FRAMES == 0:
        state.last_rects = detect_vehicles(net, output_layers, class_names, frame)

    scale_x = TILE_WIDTH / orig_w
    scale_y = TILE_HEIGHT / orig_h
    tile = cv2.resize(frame, (TILE_WIDTH, TILE_HEIGHT))

    scaled_rects = [
        (int(sx * scale_x), int(sy * scale_y), int(ex * scale_x), int(ey * scale_y))
        for (sx, sy, ex, ey) in state.last_rects
    ]

    for (startX, startY, endX, endY) in scaled_rects:
        cv2.rectangle(tile, (startX, startY), (endX, endY), BOX_COLOR, 2)

    objects = state.ct.update(scaled_rects)

    def nearest_rect(centroid):
        if not scaled_rects:
            return None
        cx, cy = centroid
        return min(
            scaled_rects,
            key=lambda r: (((r[0] + r[2]) / 2 - cx) ** 2 + ((r[1] + r[3]) / 2 - cy) ** 2),
        )

    for (object_id, centroid) in objects.items():
        to = state.trackable_objects.get(object_id, None)

        if to is None:
            to = TrackableObject(object_id, centroid)
        else:
            to.centroids.append(centroid)
            if not to.counted:
                state.total += 1
                to.counted = True

        state.trackable_objects[object_id] = to

        rect = nearest_rect(centroid)
        is_emergency = False
        if rect is not None:
            sx, sy, ex, ey = rect
            crop = tile[max(0, sy):max(0, ey), max(0, sx):max(0, ex)]
            is_emergency = state.emergency.update(object_id, crop)

        if is_emergency:
            state.emergency_ever_detected = True
            sx, sy, ex, ey = rect
            cv2.rectangle(tile, (sx, sy), (ex, ey), (255, 0, 255), 3)
            cv2.putText(tile, "EMERGENCY", (sx, max(15, sy - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        cv2.circle(tile, (int(centroid[0]), int(centroid[1])), 3, BOX_COLOR, -1)

    if state.emergency_ever_detected:
        cv2.putText(tile, "!! EMERGENCY VEHICLE - PREEMPTING SIGNAL !!", (8, TILE_HEIGHT - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    state.last_tile = tile
    return draw_overlay(tile.copy(), state)


def draw_overlay(tile, state, finished=False):
    cv2.rectangle(tile, (0, 0), (TILE_WIDTH, 34), (0, 0, 0), -1)
    status = "FINISHED" if (finished or state.finished) else "LIVE"
    if state.emergency_ever_detected:
        status += " / EMERGENCY"
    text = "{}  |  Count: {}  [{}]".format(state.label, state.total, status)
    text_color = (255, 0, 255) if state.emergency_ever_detected else (0, 255, 255)
    cv2.putText(tile, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    border_color = (255, 0, 255) if state.emergency_ever_detected else (80, 80, 80)
    cv2.rectangle(tile, (0, 0), (TILE_WIDTH - 1, TILE_HEIGHT - 1), border_color, 2)
    return tile


def run_combined(video_paths, labels, net, output_layers, class_names,
                  output_video_path=None, telemetry_path=None):
    lanes = [LaneState(path, label) for path, label in zip(video_paths, labels)]

    # Pace live playback to the source videos' own frame rate so the grid
    # window doesn't race ahead as fast as detection allows. Detection speed
    # still varies frame to frame, so the live window can look uneven; the
    # recorded output file (below) is written at a fixed fps and always
    # plays back perfectly smoothly regardless of how long detection took.
    target_fps = min(lane.fps for lane in lanes)
    frame_interval = 1.0 / target_fps

    cv2.namedWindow(GRID_WINDOW_NAME)

    writer = None
    if output_video_path is not None:
        grid_size = (TILE_WIDTH * 2, TILE_HEIGHT * 2)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, target_fps, grid_size)

    telemetry_file = None
    telemetry_writer = None
    if telemetry_path is not None:
        telemetry_file = open(telemetry_path, "w", newline="")
        telemetry_writer = csv.writer(telemetry_file)
        telemetry_writer.writerow(["elapsed_s", "lane", "count", "emergency"])

    start_time = time.perf_counter()

    while not all(lane.finished for lane in lanes):
        loop_start = time.perf_counter()

        tiles = [render_lane_tile(lane, net, output_layers, class_names) for lane in lanes]

        if telemetry_writer is not None:
            elapsed_s = round(time.perf_counter() - start_time, 2)
            for lane in lanes:
                telemetry_writer.writerow(
                    [elapsed_s, lane.label, lane.total, int(lane.emergency_ever_detected)]
                )

        top_row = np.hstack((tiles[0], tiles[1]))
        bottom_row = np.hstack((tiles[2], tiles[3]))
        grid = np.vstack((top_row, bottom_row))

        if writer is not None:
            writer.write(grid)

        cv2.imshow(GRID_WINDOW_NAME, grid)

        elapsed = time.perf_counter() - loop_start
        remaining_ms = max(1, int((frame_interval - elapsed) * 1000))

        key = cv2.waitKey(remaining_ms) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)

    for lane in lanes:
        if lane.cap is not None:
            lane.cap.release()
        emergency_note = " [EMERGENCY VEHICLE SEEN]" if lane.emergency_ever_detected else ""
        print("{}: {} vehicles detected{}".format(lane.label, lane.total, emergency_note))

    if writer is not None:
        writer.release()
        print("Saved smooth annotated video to:", output_video_path)

    if telemetry_file is not None:
        telemetry_file.close()

    cv2.destroyWindow(GRID_WINDOW_NAME)

    counts = [lane.total for lane in lanes]
    emergency_flags = [lane.emergency_ever_detected for lane in lanes]
    return counts, emergency_flags


# --- Adaptive signal timing: saturation-flow model with carry-over ---
#
# Instead of sorting counts into fixed 30/60/120s buckets (which treats a
# lane with 26 vehicles the same as one with 39), each lane's green time is
# derived from how long it actually takes to clear its queue at a realistic
# discharge rate, capped so one congested lane can't hog the whole cycle.
# Vehicles that don't fit in that cap aren't dropped - they carry over as
# real backlog into the next cycle's demand (no fabricated vehicles), and a
# lane that stays backlogged gets its cap raised a little further each
# consecutive cycle ("aging") until it clears - a bounded, linear escalation
# that's guaranteed to converge instead of compounding out of control.
SATURATION_RATE = 0.5     # vehicles a lane can clear per second of green
MIN_GREEN = 10             # seconds - floor so no lane is skipped entirely
MAX_GREEN = 60             # seconds - baseline cap per lane per cycle
CAP_ESCALATION_STEP = 15   # seconds added to the cap per consecutive backlogged cycle
ABSOLUTE_MAX_GREEN = 120   # seconds - hard ceiling even after escalation

# Emergency preemption: a lane with a detected emergency vehicle bypasses
# the cap entirely and is given enough green to clear its *entire* demand
# (live count + any backlog) in one phase, exactly like a real signal
# preemption system would hold green until the priority vehicle is through.
EMERGENCY_MIN_GREEN = 20

STATE_PATH = os.path.join(BASE_DIR, "signal_state.json")


def load_signal_state(num_lanes):
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r") as f:
                data = json.load(f)
                backlog = data.get("backlog", [])
                cycles_waited = data.get("cycles_waited", [])
                if len(backlog) == num_lanes and len(cycles_waited) == num_lanes:
                    return backlog, cycles_waited
        except (ValueError, OSError):
            pass
    return [0.0] * num_lanes, [0] * num_lanes


# Kept for backwards compatibility with anything still calling the old name.
def load_backlog(num_lanes):
    backlog, _ = load_signal_state(num_lanes)
    return backlog


def save_signal_state(backlog, cycles_waited):
    with open(STATE_PATH, "w") as f:
        json.dump({"backlog": backlog, "cycles_waited": cycles_waited}, f)


def compute_signal_plan(vehicle_counts, backlog, cycles_waited=None, emergency_flags=None):
    num_lanes = len(vehicle_counts)
    if cycles_waited is None:
        cycles_waited = [0] * num_lanes
    if emergency_flags is None:
        emergency_flags = [False] * num_lanes

    plan = []
    new_backlog = []
    new_cycles_waited = []

    for count, waiting, waited_cycles, is_emergency in zip(vehicle_counts, backlog, cycles_waited, emergency_flags):
        # Demand is always the REAL number of vehicles - live count plus
        # actual backlog. Never inflate it; only the cap below grows.
        demand = count + waiting

        if is_emergency:
            # Preempt: hold green long enough to clear the whole queue,
            # not just the usual capped share.
            green = max(EMERGENCY_MIN_GREEN, demand / SATURATION_RATE) if demand > 0 else EMERGENCY_MIN_GREEN
            served = demand
            leftover = 0.0
            next_waited_cycles = 0
        else:
            # A lane that has been backlogged for N consecutive cycles gets
            # its cap raised by CAP_ESCALATION_STEP per cycle (bounded by
            # ABSOLUTE_MAX_GREEN), so a heavy lane's window keeps widening
            # until it actually clears, instead of staying capped forever.
            effective_max_green = min(ABSOLUTE_MAX_GREEN, MAX_GREEN + waited_cycles * CAP_ESCALATION_STEP)

            ideal_green = demand / SATURATION_RATE if demand > 0 else 0
            green = max(MIN_GREEN, min(effective_max_green, ideal_green)) if demand > 0 else MIN_GREEN

            capacity = green * SATURATION_RATE
            served = min(demand, capacity)
            leftover = max(0.0, demand - capacity)
            next_waited_cycles = waited_cycles + 1 if leftover > 0 else 0

        plan.append({
            "count": count,
            "backlog_in": round(waiting, 1),
            "demand": round(demand, 1),
            "green": round(green, 1),
            "served": round(served, 1),
            "leftover": round(leftover, 1),
            "cycles_waited": next_waited_cycles,
            "emergency": bool(is_emergency),
        })
        new_backlog.append(leftover)
        new_cycles_waited.append(next_waited_cycles)

    return plan, new_backlog, new_cycles_waited


def show_summary_window(plan):
    lane_labels = ["Lane {}".format(i + 1) for i in range(len(plan))]
    greens = [p["green"] for p in plan]

    panel_height = 480
    panel_width = 760
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

    cv2.putText(panel, "Adaptive Traffic Signal - Summary", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    row_y = 90
    for label, p in zip(lane_labels, plan):
        tag = "  [EMERGENCY PREEMPTION]" if p.get("emergency") else ""
        line1 = "{}: detected {}  (+{} carried over) -> demand {}{}".format(
            label, p["count"], p["backlog_in"], p["demand"], tag)
        line2 = "  green {:.1f}s  |  served {:.1f}  |  carries over {:.1f}".format(
            p["green"], p["served"], p["leftover"])
        line_color = (255, 0, 255) if p.get("emergency") else (255, 255, 255)
        bar_color = (255, 0, 255) if p.get("emergency") else (0, 200, 0)
        cv2.putText(panel, line1, (30, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_color, 1)
        cv2.putText(panel, line2, (30, row_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 255, 150), 1)
        bar_len = int((p["green"] / max(greens)) * 500) if max(greens) > 0 else 0
        cv2.rectangle(panel, (30, row_y + 32), (30 + bar_len, row_y + 48), bar_color, -1)
        row_y += 80

    cv2.putText(panel, "Total cycle time: {:.1f}s".format(sum(greens)), (30, row_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)
    cv2.putText(panel, "Press any key to close", (30, panel_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow("Traffic Signal Summary", panel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    class_names = load_class_names()
    net, output_layers = load_network()

    video_files = ["1.mp4", "2.mp4", "3.mp4", "4.mp4"]
    labels = ["Lane 1", "Lane 2", "Lane 3", "Lane 4"]
    video_paths = [os.path.join(VIDEOS_DIR, f) for f in video_files]

    outputs_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    output_video_path = os.path.join(outputs_dir, "traffic_detection_output.mp4")
    telemetry_path = os.path.join(outputs_dir, "telemetry.csv")

    vehicle_counts, emergency_flags = run_combined(
        video_paths, labels, net, output_layers, class_names,
        output_video_path, telemetry_path,
    )

    out_path = os.path.join(BASE_DIR, "out.txt")
    with open(out_path, "w") as f:
        for count in vehicle_counts:
            f.write("{}\n".format(count))

    backlog, cycles_waited = load_signal_state(len(vehicle_counts))
    plan, new_backlog, new_cycles_waited = compute_signal_plan(vehicle_counts, backlog, cycles_waited, emergency_flags)
    save_signal_state(new_backlog, new_cycles_waited)

    print("Vehicle counts per lane:", vehicle_counts)
    for label, p in zip(labels, plan):
        tag = " [EMERGENCY PREEMPTION]" if p["emergency"] else ""
        print("{}: demand={} green={:.1f}s served={:.1f} carries_over={:.1f}{}".format(
            label, p["demand"], p["green"], p["served"], p["leftover"], tag))
    print("Total cycle time: {:.1f}s".format(sum(p["green"] for p in plan)))

    latest_run_path = os.path.join(outputs_dir, "latest_run.json")
    with open(latest_run_path, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "labels": labels,
            "vehicle_counts": vehicle_counts,
            "emergency_flags": emergency_flags,
            "plan": plan,
            "total_cycle_time": sum(p["green"] for p in plan),
        }, f, indent=2)

    show_summary_window(plan)


if __name__ == "__main__":
    main()
