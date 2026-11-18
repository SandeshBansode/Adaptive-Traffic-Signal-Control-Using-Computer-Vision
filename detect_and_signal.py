"""
Adaptive Traffic Signal Control - live camera demo

Each of the four videos is one approach to an intersection. The program
works like a real camera-actuated signal:

  1. YOLOv4-tiny (OpenCV DNN) detects vehicles in every approach.
  2. Only vehicles standing between that camera's two road-edge lines are
     considered (road_roi.json - draw the lines with calibrate_roi.py; the
     tool opens automatically for a camera without lines), so pavements,
     side roads and cars parked off the road are ignored. A vehicle inside
     the lines that doesn't move while its traffic has green is treated as
     parked and ignored too.
  3. The number of vehicles inside the zone is that approach's queue.
  4. controller.py turns the four queues into signal lights following
     real-world rules: opposing approaches (North + South, East + West)
     share the green, and every change goes through a yellow and an
     all-red interval sized from the speed limit and junction width.
  5. The lights drive the videos: a RED approach's video is paused (its
     traffic is stopped and its queue is frozen), a GREEN approach's video
     plays (its traffic moves), and the controller keeps re-reading the
     queues to decide when to switch and which phase goes next.

Usage:
    python detect_and_signal.py                        # adaptive, two-phase
    python detect_and_signal.py --mode fixed           # fixed-timer baseline, same videos
    python detect_and_signal.py --phasing four-phase   # one approach at a time
    python detect_and_signal.py --pedestrians          # pedestrian-safe minimum green
    python detect_and_signal.py --loop --fast          # loop the footage, run unthrottled

Keys: P pause, F toggle fast mode, Q quit.
"""

import argparse
import csv
import json
import os
import statistics
import time
from collections import deque

import cv2
import numpy as np

import ui
from controller import (
    DEFAULT_INTERSECTION_WIDTH_M,
    DEFAULT_SPEED_KMH,
    FIXED_GREEN,
    GREEN,
    MAX_GREEN,
    MIN_GREEN,
    RED,
    SATURATION_RATE,
    YELLOW,
    AdaptiveController,
    FixedTimeController,
    change_intervals,
    pedestrian_min_green,
    single_phase_plan,
    two_phase_plan,
)
from emergency import EmergencyDetector
from road_roi import RoadROI, load_config
from tracking.centroidtracker import CentroidTracker
from tracking.trackableobject import TrackableObject

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


def load_class_names():
    with open(NAMES_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_network():
    if not hasattr(cv2.dnn, "readNetFromDarknet"):
        # OpenCV 5 removed the Darknet importer that YOLOv4-tiny needs.
        raise SystemExit(
            "OpenCV {} cannot load Darknet (YOLO .cfg/.weights) models. "
            "Install OpenCV 4.x: pip install \"opencv-python>=4.5.4,<5\"".format(cv2.__version__)
        )
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
GRID_WINDOW_NAME = "Adaptive Traffic Signal - Intersection"
SUMMARY_WINDOW_NAME = "Traffic Signal Report"

DEFAULT_VIDEOS = ["1.mp4", "2.mp4", "3.mp4", "4.mp4"]
# Approach order matches the SUMO network and the two-phase plan:
# 0 + 2 (North + South) share a phase, as do 1 + 3 (East + West).
DEFAULT_LABELS = ["North", "East", "South", "West"]

# The queue is the median of the last few detection counts, so one missed
# or doubled detection doesn't flip the signal decision.
QUEUE_SMOOTHING = 5
TELEMETRY_INTERVAL_S = 0.5

# A vehicle has "moved" once its centre travels this far (tile pixels) -
# well above detection jitter. One that stays put for PARKED_AFTER_S
# seconds of *playing* footage (its traffic had green and still didn't
# move) is parked: it's left out of the queue and the count. Paused (red)
# footage doesn't advance this clock, so cars queued at a red are safe.
MOVE_THRESHOLD_PX = 12
PARKED_AFTER_S = 20


class LaneCamera:
    """One approach: its video, road zone, detections, tracker and stats."""

    def __init__(self, video_path, label, roi, loop=False):
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.label = label
        self.roi = roi
        self.loop = loop

        cap = cv2.VideoCapture(video_path) if os.path.exists(video_path) else None
        if cap is not None and not cap.isOpened():
            cap = None
        self.cap = cap
        self.has_video = cap is not None
        self.finished = not self.has_video
        self.fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else 30.0
        if not self.fps or self.fps <= 1:
            self.fps = 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap is not None else 0
        self.frame_idx = 0
        self.full_frame = None
        self.frame = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)

        self.boxes = []                     # (rect, on_road, object_id, is_emergency) in tile coords
        self.queue_history = deque(maxlen=QUEUE_SMOOTHING)
        self._reset_tracking()
        self.counted = 0
        self.emergency_active = False

        self.green_time = 0.0
        self.max_queue = 0
        self.queue_integral = 0.0

    def _reset_tracking(self):
        self.ct = CentroidTracker(maxDisappeared=15, maxDistance=75)
        self.trackable_objects = {}
        self.emergency = EmergencyDetector()

    @property
    def queue(self):
        if self.finished or not self.queue_history:
            return 0
        return int(round(statistics.median(self.queue_history)))

    @property
    def progress(self):
        if self.finished:
            return 1.0
        return self.frame_idx / self.total_frames if self.total_frames > 0 else 0.0

    def read_next(self):
        """Advances the video one frame; returns False once it has ended."""
        if self.finished:
            return False
        ok, frame = self.cap.read()
        if not ok and self.loop and self.frame_idx > 0:
            # Treat the clip as a continuous stream of arriving traffic
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if ok:
                self.frame_idx = 0
                self._reset_tracking()
        if not ok:
            self.finished = True
            self.cap.release()
            self.boxes = []
            self.queue_history.clear()
            self.emergency_active = False
            return False
        self.frame_idx += 1
        self.full_frame = frame
        self.frame = cv2.resize(frame, (TILE_WIDTH, TILE_HEIGHT))
        return True

    def analyze(self, net, output_layers, class_names):
        """Detects vehicles on the current frame, keeps the ones inside the
        road zone, and updates the queue, tracker, count and emergency flag."""
        if self.full_frame is None:
            return
        orig_h, orig_w = self.full_frame.shape[:2]
        scale_x = TILE_WIDTH / orig_w
        scale_y = TILE_HEIGHT / orig_h
        rects = [
            (int(sx * scale_x), int(sy * scale_y), int(ex * scale_x), int(ey * scale_y))
            for (sx, sy, ex, ey) in detect_vehicles(net, output_layers, class_names, self.full_frame)
        ]
        on_road = [self.roi.contains_rect(r, TILE_WIDTH, TILE_HEIGHT) if self.roi else True for r in rects]
        road_rects = [r for r, inside in zip(rects, on_road) if inside]

        objects = self.ct.update(road_rects)

        def nearest_rect(centroid):
            cx, cy = centroid
            return min(road_rects,
                       key=lambda r: ((r[0] + r[2]) / 2 - cx) ** 2 + ((r[1] + r[3]) / 2 - cy) ** 2)

        identified = {}
        emergency_active = False
        parked_after_frames = PARKED_AFTER_S * self.fps
        for object_id, centroid in objects.items():
            to = self.trackable_objects.get(object_id)
            if to is None:
                to = TrackableObject(object_id, centroid)
                to.frames.append(self.frame_idx)
                self.trackable_objects[object_id] = to
            elif self.ct.disappeared.get(object_id, 0) == 0:
                to.centroids.append(centroid)
                to.frames.append(self.frame_idx)
                self._update_motion(to, parked_after_frames)

            # Only objects matched to a box this frame - an unseen object's
            # "nearest" box belongs to another vehicle.
            if not road_rects or self.ct.disappeared.get(object_id, 0) != 0:
                continue
            rect = nearest_rect(centroid)
            sx, sy, ex, ey = rect
            crop = self.frame[max(0, sy):max(0, ey), max(0, sx):max(0, ex)]
            is_emergency = self.emergency.update(object_id, crop) and not to.parked
            emergency_active = emergency_active or is_emergency
            identified[rect] = (object_id, is_emergency, to.parked)

        for object_id in list(self.trackable_objects):
            if object_id not in objects:
                del self.trackable_objects[object_id]
                self.emergency.forget(object_id)

        self.emergency_active = emergency_active
        self.boxes = []
        for rect, inside in zip(rects, on_road):
            object_id, is_emergency, parked = identified.get(rect, (None, False, False)) if inside \
                else (None, False, False)
            self.boxes.append((rect, inside, object_id, is_emergency, parked))
        self.queue_history.append(sum(1 for b in self.boxes if b[1] and not b[4]))

    def _update_motion(self, to, parked_after_frames):
        """Counts a vehicle once it actually travels, and flags it as parked
        if it hasn't moved over the last PARKED_AFTER_S seconds of footage."""
        # Positions are compared with averages, so box jitter around a
        # stationary car doesn't look like movement.
        def mean(points):
            return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))

        def dist(a, b):
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

        if not to.counted and dist(to.centroids[-1], mean(to.centroids[:5])) >= MOVE_THRESHOLD_PX:
            self.counted += 1
            to.counted = True

        window_start = self.frame_idx - parked_after_frames
        if to.frames[0] > window_start:
            to.parked = False       # not watched long enough to judge yet
            return
        # Keep one sample from before the window as the reference point
        while len(to.frames) > 2 and to.frames[1] <= window_start:
            to.frames.pop(0)
            to.centroids.pop(0)
        centre = mean(to.centroids)
        to.parked = all(dist(c, centre) < MOVE_THRESHOLD_PX for c in to.centroids)


def light_label(lane, light, ctrl, index):
    if lane.finished:
        return "CLEARED" if lane.has_video else "NO VIDEO"
    if light == GREEN:
        remaining = ctrl.time_remaining()
        return "GREEN {:.0f}s".format(remaining) if remaining > 0 else "GREEN (holding)"
    if light == YELLOW:
        return "YELLOW"
    wait = ctrl.lane_red_wait(index)
    return "RED  waited {:.0f}s".format(wait) if wait >= 1 else "RED"


def phase_names(ctrl, labels):
    return [" + ".join(labels[lane] for lane in phase) for phase in ctrl.phases]


def phase_text(ctrl, labels):
    if ctrl.green_phase is None:
        return "ALL RED"
    name = phase_names(ctrl, labels)[ctrl.green_phase]
    if ctrl.phase == GREEN:
        return "{} GREEN  {:.0f} / {:.0f} s".format(name, ctrl.phase_elapsed, ctrl.planned_green)
    return "{} YELLOW".format(name)


def decision_text(ctrl, labels):
    if not ctrl.phase_log:
        return "Waiting for the first decision..."
    names = phase_names(ctrl, labels)
    last = ctrl.phase_log[-1]
    s = "Decision: {} -> green by {} ({} queued, waited {:.0f}s, planned {:.0f}s)".format(
        names[last["phase"]], last["choice"], last["queue_at_start"], last["waited"], last["planned"])
    finished = [e for e in ctrl.phase_log if e["end_reason"]]
    if finished:
        prev = finished[-1]
        s += "    Last: {} ended by {} after {:.0f}s".format(
            names[prev["phase"]], prev["end_reason"], prev["green"])
    return s


def render_grid(lanes, lights, ctrl, labels, mode, fps, fast, paused=False):
    tiles = []
    for i, (lane, light) in enumerate(zip(lanes, lights)):
        green_progress = None
        if light == GREEN and i in ctrl.green_lanes and ctrl.planned_green > 0:
            green_progress = ctrl.phase_elapsed / ctrl.planned_green
        notice = None
        if not lane.has_video:
            notice = "Video not found: {}".format(lane.video_name)
        elif lane.roi is None:
            notice = "No road lines: run calibrate_roi.py"
        tiles.append(ui.lane_tile(
            lane.frame, lane.label, light, light_label(lane, light, ctrl, i), lane.queue, lane.counted,
            lane.boxes, lane.roi, lane.progress, green_progress, lane.finished, lane.emergency_active,
            flash_phase=int(ctrl.time * 4), notice=notice,
        ))
    grid = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
    if paused:
        grid = ui.paused_overlay(grid)
    bar = ui.status_bar(grid.shape[1], ctrl.time, mode, phase_text(ctrl, labels),
                        decision_text(ctrl, labels), fps, fast, paused)
    return np.vstack((grid, bar))


def lane_summary(lane, index, ctrl):
    phases = [e for e in ctrl.phase_log if index in e["lanes"]]
    return {
        "label": lane.label,
        "video": lane.video_name,
        "counted": lane.counted,
        "green_time": round(lane.green_time, 1),
        "phases": len(phases),
        "avg_wait": round(sum(e["waited"] for e in phases) / len(phases), 1) if phases else None,
        "max_queue": lane.max_queue,
        "avg_queue": round(lane.queue_integral / ctrl.time, 2) if ctrl.time > 0 else 0.0,
        "road_lines": lane.roi is not None,
    }


def build_controller(args, num_lanes):
    phases = two_phase_plan(num_lanes) if args.phasing == "two-phase" else single_phase_plan(num_lanes)
    yellow, all_red = change_intervals(args.speed_limit, args.intersection_width)
    if args.mode == "fixed":
        return FixedTimeController(num_lanes, phases, green_time=args.fixed_green,
                                   yellow_time=yellow, all_red_time=all_red)
    min_green = args.min_green
    if args.pedestrians:
        min_green = max(min_green, pedestrian_min_green(args.intersection_width))
    return AdaptiveController(num_lanes, phases, min_green=min_green, max_green=args.max_green,
                              yellow_time=yellow, all_red_time=all_red)


def run(args):
    class_names = load_class_names()
    net, output_layers = load_network()

    video_paths = [p if os.path.isabs(p) or os.path.exists(p) else os.path.join(VIDEOS_DIR, p)
                   for p in args.videos]
    labels = DEFAULT_LABELS[:len(video_paths)]
    rois = load_config()

    missing = [p for p in video_paths if os.path.exists(p) and RoadROI.from_config(os.path.basename(p), rois) is None]
    if missing and not (args.skip_calibration or args.headless):
        # Counting needs the road edges - draw them before the run starts
        import calibrate_roi
        print("No road lines yet for: {} - opening the calibration tool.".format(
            ", ".join(os.path.basename(p) for p in missing)))
        calibrate_roi.calibrate(missing, 0)
        rois = load_config()
    lanes = [LaneCamera(path, label, RoadROI.from_config(os.path.basename(path), rois), args.loop)
             for path, label in zip(video_paths, labels)]

    for lane in lanes:
        if not lane.has_video:
            print("Warning: {} video not found ({})".format(lane.label, lane.video_path))
        elif lane.roi is None:
            print("Warning: no road lines for {} ({}) - counting the whole frame. "
                  "Run: python calibrate_roi.py".format(lane.label, lane.video_name))

    ctrl = build_controller(args, len(lanes))
    print("Signal plan: {} | phases: {} | yellow {:.1f}s, all-red {:.1f}s{}".format(
        args.mode, ", ".join(phase_names(ctrl, labels)), ctrl.yellow_time, ctrl.all_red_time,
        " | min green {:.1f}s (pedestrians)".format(ctrl.min_green) if args.pedestrians and args.mode != "fixed"
        else ""))

    # The signal clock runs on video time: one tick = one video frame, so
    # timings are exact no matter how fast detection runs on this machine.
    fps = min([lane.fps for lane in lanes if lane.has_video] or [30.0])
    dt = 1.0 / fps

    for lane in lanes:
        if lane.read_next():
            lane.analyze(net, output_layers, class_names)

    outputs_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    writer = None
    if not args.no_record:
        grid_size = (TILE_WIDTH * 2, TILE_HEIGHT * 2 + ui.STATUS_BAR_HEIGHT)
        writer = cv2.VideoWriter(os.path.join(outputs_dir, "traffic_detection_output.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, grid_size)
    telemetry_file = open(os.path.join(outputs_dir, "telemetry.csv"), "w", newline="")
    telemetry = csv.writer(telemetry_file)
    telemetry.writerow(["t", "lane", "light", "queue", "counted"])
    next_telemetry = 0.0

    mode_label = "{} {}".format(args.mode, args.phasing)
    if not args.headless:
        cv2.namedWindow(GRID_WINDOW_NAME)
    lights = [RED] * len(lanes)
    fast = args.fast or args.headless
    render_fps = 0.0

    try:
        while True:
            loop_start = time.perf_counter()

            # Traffic on the green/yellow approach moves; red approaches stay paused
            for lane, light in zip(lanes, lights):
                if light != RED and lane.read_next() and lane.frame_idx % DETECT_EVERY_N_FRAMES == 0:
                    lane.analyze(net, output_layers, class_names)

            queues = [lane.queue for lane in lanes]
            lights = ctrl.update(dt, queues, [lane.emergency_active for lane in lanes],
                                 [not lane.finished for lane in lanes])

            for lane, light, queue in zip(lanes, lights, queues):
                if light == GREEN:
                    lane.green_time += dt
                lane.max_queue = max(lane.max_queue, queue)
                lane.queue_integral += queue * dt

            if ctrl.time >= next_telemetry:
                for lane, light, queue in zip(lanes, lights, queues):
                    telemetry.writerow([round(ctrl.time, 2), lane.label, light, queue, lane.counted])
                next_telemetry += TELEMETRY_INTERVAL_S

            grid = render_grid(lanes, lights, ctrl, labels, mode_label, render_fps, fast)
            if writer is not None:
                writer.write(grid)
            if all(lane.finished for lane in lanes):
                break
            if args.duration and ctrl.time >= args.duration:
                break
            if args.headless:
                continue
            cv2.imshow(GRID_WINDOW_NAME, grid)

            elapsed = time.perf_counter() - loop_start
            wait_ms = 1 if fast else max(1, int((dt - elapsed) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f"):
                fast = not fast
            elif key == ord("p"):
                cv2.imshow(GRID_WINDOW_NAME, render_grid(lanes, lights, ctrl, labels, mode_label,
                                                         render_fps, fast, paused=True))
                if cv2.waitKey(0) & 0xFF == ord("q"):
                    break

            loop_fps = 1.0 / max(1e-6, time.perf_counter() - loop_start)
            render_fps = loop_fps if render_fps == 0.0 else 0.9 * render_fps + 0.1 * loop_fps
    finally:
        for lane in lanes:
            if lane.cap is not None:
                lane.cap.release()
        if writer is not None:
            writer.release()
        telemetry_file.close()
        if not args.headless:
            cv2.destroyWindow(GRID_WINDOW_NAME)

    return lanes, ctrl, labels


def save_results(lanes, ctrl, labels, args):
    mode = args.mode
    outputs_dir = os.path.join(BASE_DIR, "outputs")
    stats = [lane_summary(lane, i, ctrl) for i, lane in enumerate(lanes)]
    names = phase_names(ctrl, labels)
    phases = [dict(e, label=names[e["phase"]]) for e in ctrl.phase_log]

    # Per-approach vehicle counts, used by simulate_comparison.py and
    # sumo_sim/generate_routes.py as each approach's relative demand.
    with open(os.path.join(BASE_DIR, "out.txt"), "w") as f:
        for lane in lanes:
            f.write("{}\n".format(lane.counted))

    with open(os.path.join(outputs_dir, "latest_run.json"), "w") as f:
        json.dump({
            "timestamp": time.time(),
            "mode": mode,
            "labels": labels,
            "sim_time": round(ctrl.time, 2),
            "phase_plan": [{"name": name, "lanes": phase} for name, phase in zip(names, ctrl.phases)],
            "settings": {
                "phasing": args.phasing,
                "min_green": getattr(ctrl, "min_green", None),
                "max_green": getattr(ctrl, "max_green", None),
                "fixed_green": getattr(ctrl, "green_time", None),
                "saturation_rate": SATURATION_RATE,
                "yellow": ctrl.yellow_time,
                "all_red": ctrl.all_red_time,
                "speed_limit_kmh": args.speed_limit,
                "intersection_width_m": args.intersection_width,
                "pedestrians": args.pedestrians,
            },
            "lanes": stats,
            "phases": phases,
        }, f, indent=2)

    print("\n{} control - {:.0f} s simulated, {} greens".format(mode.capitalize(), ctrl.time,
                                                                    len(ctrl.phase_log)))
    for s in stats:
        print("{:5s}: counted {:3d} | green {:5.1f}s over {} phase(s) | avg wait {} | max queue {}".format(
            s["label"], s["counted"], s["green_time"], s["phases"],
            "{:.0f}s".format(s["avg_wait"]) if s["avg_wait"] is not None else "-", s["max_queue"]))
    return stats, phases


def show_summary_window(labels, stats, phases, sim_time, mode, phasing, headless=False):
    panel = ui.summary_panel(labels, stats, phases, sim_time, mode, phasing)
    path = os.path.join(BASE_DIR, "outputs", "summary.png")
    cv2.imwrite(path, panel)
    if headless:
        print("Summary saved to {}".format(path))
        return
    cv2.imshow(SUMMARY_WINDOW_NAME, panel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Camera-actuated adaptive traffic signal demo")
    parser.add_argument("--mode", choices=["adaptive", "fixed"], default="adaptive",
                        help="adaptive control, or the fixed-timer baseline for comparison")
    parser.add_argument("--videos", nargs=4, default=DEFAULT_VIDEOS, metavar="VIDEO",
                        help="four approach videos (default: videos/1.mp4 .. 4.mp4)")
    parser.add_argument("--loop", action="store_true",
                        help="loop each video so traffic keeps arriving (stop with Q or --duration)")
    parser.add_argument("--duration", type=float, default=None,
                        help="stop after this many simulated seconds")
    parser.add_argument("--fast", action="store_true", help="don't throttle to real time (toggle with F)")
    parser.add_argument("--no-record", action="store_true", help="don't save outputs/traffic_detection_output.mp4")
    parser.add_argument("--min-green", type=float, default=MIN_GREEN)
    parser.add_argument("--max-green", type=float, default=MAX_GREEN)
    parser.add_argument("--fixed-green", type=float, default=FIXED_GREEN)
    parser.add_argument("--phasing", choices=["two-phase", "four-phase"], default="two-phase",
                        help="two-phase: North+South then East+West (default); "
                             "four-phase: one approach at a time")
    parser.add_argument("--speed-limit", type=float, default=DEFAULT_SPEED_KMH,
                        help="approach speed in km/h - sets the yellow and all-red times")
    parser.add_argument("--intersection-width", type=float, default=DEFAULT_INTERSECTION_WIDTH_M,
                        help="junction width in metres - sets the all-red (and pedestrian) time")
    parser.add_argument("--pedestrians", action="store_true",
                        help="never give less green than a pedestrian needs to cross")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="don't open the road-line tool for cameras without lines")
    parser.add_argument("--headless", action="store_true",
                        help="no windows (servers, Google Colab): run at full speed and save "
                             "the video and outputs/summary.png")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    lanes, ctrl, labels = run(args)
    stats, phases = save_results(lanes, ctrl, labels, args)
    show_summary_window(labels, stats, phases, ctrl.time, args.mode, args.phasing, args.headless)


if __name__ == "__main__":
    main()
