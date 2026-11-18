"""
Road counting zone for each camera.

Each video gets two lines drawn along the two edges of the road (set them
with calibrate_roi.py). Each line is extended across the whole frame and
the counting zone is the area between the two lines,
and a vehicle only counts if its ground point - the bottom-centre of its
box, where the wheels touch the road - is inside that zone. Parked cars,
vehicles on side roads and anything outside the lines are ignored.

Lines are stored in road_roi.json in normalized (0-1) coordinates, keyed by
video file name, so they work at any display or detection resolution:

    {"1.mp4": {"left": [[x, y], [x, y]], "right": [[x, y], [x, y]]}}
"""

import json
import os

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "road_roi.json")


def load_config(path=CONFIG_PATH):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_roi(video_name, roi, path=CONFIG_PATH):
    config = load_config(path)
    config[video_name] = {"left": roi.left, "right": roi.right}
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def ground_point(rect):
    """Bottom-centre of a (startX, startY, endX, endY) box."""
    sx, sy, ex, ey = rect
    return ((sx + ex) / 2.0, float(ey))


def _side(a, b, p):
    """Sign of p relative to the infinite line through a and b."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _clip(polygon, a, b, sign):
    """Keeps the part of a convex polygon on the `sign` side of line a-b
    (Sutherland-Hodgman, one edge)."""
    out = []
    for i, cur in enumerate(polygon):
        prev = polygon[i - 1]
        cur_in = _side(a, b, cur) * sign >= 0
        prev_in = _side(a, b, prev) * sign >= 0
        if cur_in != prev_in:
            d_prev, d_cur = _side(a, b, prev), _side(a, b, cur)
            t = d_prev / (d_prev - d_cur)
            out.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
        if cur_in:
            out.append(cur)
    return out


class RoadROI:
    """Area between two road-edge lines. Each line is extended across the
    whole frame, so the zone covers the full length of the road in view -
    including where the edges run off the sides of the image."""

    def __init__(self, left, right):
        left = [tuple(map(float, p)) for p in left]
        right = [tuple(map(float, p)) for p in right]
        if len(left) != 2 or len(right) != 2:
            raise ValueError("each road edge needs exactly two points")
        if left[0] == left[1] or right[0] == right[1]:
            raise ValueError("a road edge needs two different points")
        # The road is on the side of each line where the other line is.
        # (Sign of the cross product is unchanged by scaling x and y, so it
        # can be worked out once in normalized coordinates.)
        mid_left = ((left[0][0] + left[1][0]) / 2, (left[0][1] + left[1][1]) / 2)
        mid_right = ((right[0][0] + right[1][0]) / 2, (right[0][1] + right[1][1]) / 2)
        self._left_sign = _side(left[0], left[1], mid_right)
        self._right_sign = _side(right[0], right[1], mid_left)
        if self._left_sign == 0 or self._right_sign == 0:
            raise ValueError("the two road edges must be separate lines")
        self.left = [list(p) for p in left]
        self.right = [list(p) for p in right]

    @classmethod
    def from_config(cls, video_name, config):
        entry = config.get(video_name)
        if not entry:
            return None
        try:
            return cls(entry["left"], entry["right"])
        except (KeyError, TypeError, ValueError):
            return None

    def _contains_normalized(self, p):
        return (_side(self.left[0], self.left[1], p) * self._left_sign >= 0
                and _side(self.right[0], self.right[1], p) * self._right_sign >= 0)

    def polygon(self, width, height):
        """Zone outline (clipped to the frame) in pixel coordinates."""
        poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        poly = _clip(poly, self.left[0], self.left[1], 1 if self._left_sign > 0 else -1)
        poly = _clip(poly, self.right[0], self.right[1], 1 if self._right_sign > 0 else -1)
        return np.array([[int(round(x * width)), int(round(y * height))] for x, y in poly], dtype=np.int32)

    def edge_lines(self, width, height):
        """The two road-edge lines extended to the frame border, in pixels."""
        lines = []
        for a, b in (self.left, self.right):
            # Clip a long segment through a-b to the frame
            dx, dy = b[0] - a[0], b[1] - a[1]
            far_a = (a[0] - 10 * dx, a[1] - 10 * dy)
            far_b = (b[0] + 10 * dx, b[1] + 10 * dy)
            ok, p1, p2 = cv2.clipLine((0, 0, width, height),
                                      (int(far_a[0] * width), int(far_a[1] * height)),
                                      (int(far_b[0] * width), int(far_b[1] * height)))
            if ok:
                lines.append((p1, p2))
        return lines

    def contains(self, point, width, height):
        return self._contains_normalized((point[0] / width, point[1] / height))

    def contains_rect(self, rect, width, height):
        return self.contains(ground_point(rect), width, height)
