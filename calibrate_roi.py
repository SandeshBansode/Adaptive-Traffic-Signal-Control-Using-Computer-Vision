"""
Draw the two road-edge lines for each camera.

Shows a frame from each video. Click four points:
    1-2  two points on one edge of the road
    3-4  two points on the other edge
Each pair defines a line that is extended across the whole frame. Only
vehicles whose wheels (bottom-centre of the box) are between the two lines
are counted - pick points far apart along each edge for an accurate line.

Draw the lines around the lanes that lead INTO the intersection - leave
out the opposite carriageway (that traffic is leaving), pavements, verges
and parking bays. Once both lines are drawn, the vehicles YOLO finds on
that frame are previewed: green = counted, grey = ignored.

Keys:
    S / Enter   save and go to the next video
    U           undo last point
    R           reset this video's lines
    C           copy the lines from the previous camera (same view)
    N           skip this video without saving
    Q / Esc     quit

Usage:
    python calibrate_roi.py                    # videos/1.mp4 .. 4.mp4
    python calibrate_roi.py videos/2.mp4       # just one video
    python calibrate_roi.py --frame 60         # calibrate on frame 60 instead of the first
"""

import argparse
import os

import cv2

import ui
from detect_and_signal import DEFAULT_VIDEOS, VIDEOS_DIR, detect_vehicles, load_class_names, load_network
from road_roi import CONFIG_PATH, RoadROI, load_config, save_roi

MAX_DISPLAY_WIDTH = 1280
WINDOW_NAME = "Calibrate road lines"


def read_frame(path, frame_number):
    cap = cv2.VideoCapture(path)
    if frame_number > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


class Calibrator:
    def __init__(self, frame, existing, detections):
        h, w = frame.shape[:2]
        self.scale = min(1.0, MAX_DISPLAY_WIDTH / w)
        self.width, self.height = int(w * self.scale), int(h * self.scale)
        self.frame = cv2.resize(frame, (self.width, self.height))
        self.detections = [tuple(int(v * self.scale) for v in r) for r in detections]
        # Points in normalized coordinates
        self.points = []
        if existing is not None:
            self.points = existing.left + existing.right

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append([x / self.width, y / self.height])

    def roi(self):
        return RoadROI(self.points[:2], self.points[2:]) if len(self.points) == 4 else None

    def render(self, video_name, index, total):
        img = self.frame.copy()
        px = [(int(x * self.width), int(y * self.height)) for x, y in self.points]

        roi = self.roi()
        if roi is not None:
            ui.road_zone(img, roi, "green")
            inside = 0
            for rect in self.detections:
                if roi.contains_rect(rect, self.width, self.height):
                    ui.vehicle_box(img, rect)
                    inside += 1
                else:
                    ui.off_road_box(img, rect)
            status = "{} of {} detected vehicles inside the road zone".format(inside, len(self.detections))
        else:
            for i in range(0, len(px) - 1, 2):
                cv2.line(img, px[i], px[i + 1], ui.ROAD_LINE, 2, cv2.LINE_AA)
            steps = ["click a point on one edge of the lanes heading INTO the junction",
                     "another point on the same edge (far from the first)",
                     "click a point on the other edge of those lanes",
                     "another point on that edge"]
            status = "Point {}/4 - {}".format(len(self.points) + 1, steps[len(self.points)])

        for i, p in enumerate(px):
            cv2.circle(img, p, 6, ui.ROAD_LINE, -1, cv2.LINE_AA)
            ui.text(img, str(i + 1), (p[0] + 8, p[1] - 8), 0.55, ui.ROAD_LINE)

        ui.blend_rect(img, (0, 0), (self.width, 64), ui.BG, 0.8)
        ui.text(img, "{}  ({}/{})".format(video_name, index + 1, total), (12, 26), 0.65, ui.TEXT, 1, ui.FONT_BOLD)
        ui.text(img, status, (12, 52), 0.5, ui.AMBER if roi is None else ui.GREEN)
        hint = "[S] save  [U] undo  [R] reset  [C] copy previous  [N] skip  [Q] quit"
        ui.text(img, hint, (self.width - ui.text_size(hint, 0.45)[0] - 12, 26), 0.45, ui.MUTED)
        return img


def calibrate(paths, frame_number):
    config = load_config()
    class_names = load_class_names()
    net, output_layers = load_network()

    cv2.namedWindow(WINDOW_NAME)
    saved = []
    previous = next((RoadROI.from_config(n, config) for n in config
                     if RoadROI.from_config(n, config) is not None), None)
    for index, path in enumerate(paths):
        name = os.path.basename(path)
        frame = read_frame(path, frame_number)
        if frame is None:
            print("Skipping {}: could not read a frame".format(path))
            continue

        detections = detect_vehicles(net, output_layers, class_names, frame)
        cal = Calibrator(frame, RoadROI.from_config(name, config), detections)
        cv2.setMouseCallback(WINDOW_NAME, cal.on_mouse)

        while True:
            cv2.imshow(WINDOW_NAME, cal.render(name, index, len(paths)))
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return saved
            if key == ord("u") and cal.points:
                cal.points.pop()
            elif key == ord("r"):
                cal.points = []
            elif key == ord("c") and previous is not None:
                cal.points = [list(p) for p in previous.left + previous.right]
            elif key == ord("n"):
                break
            elif key in (ord("s"), 13, 10):
                roi = cal.roi()
                if roi is None:
                    continue
                save_roi(name, roi)
                saved.append(name)
                previous = roi
                print("Saved road lines for {} to {}".format(name, CONFIG_PATH))
                break

    cv2.destroyAllWindows()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Draw the road-edge counting lines for each camera")
    parser.add_argument("videos", nargs="*", help="videos to calibrate (default: videos/1.mp4 .. 4.mp4)")
    parser.add_argument("--frame", type=int, default=0, help="frame number to calibrate on")
    args = parser.parse_args()

    paths = args.videos or [os.path.join(VIDEOS_DIR, v) for v in DEFAULT_VIDEOS]
    saved = calibrate(paths, args.frame)
    print("Done - {} video(s) calibrated.".format(len(saved)))


if __name__ == "__main__":
    main()
