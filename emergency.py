"""
Emergency vehicle detection heuristic.

COCO (what YOLOv4-tiny is trained on) has no "ambulance" / "fire truck"
class - those are just "car" or "truck" to the detector. Instead of a
class label, this looks for the thing that actually distinguishes an
emergency vehicle on camera: a red/blue light bar that alternates color
rapidly (a "flash"). For each tracked vehicle box we sample the dominant
siren-like color on every frame and keep a short rolling history; if that
history flips between red-dominant and blue-dominant several times in a
short window, the vehicle is flagged as an emergency vehicle.

This is a heuristic, not a trained classifier - it's meant to demonstrate
the preemption *logic* (Module 3 of the project brief). See
test_emergency.py for a synthetic (non-video) test of the color/flash
logic itself.
"""

from collections import deque

import cv2
import numpy as np

# Fraction of pixels in a box's upper region that must be saturated
# red/blue for that frame to count as "signal color present".
COLOR_FRACTION_THRESHOLD = 0.05

# How many recent frames we remember per tracked vehicle.
HISTORY_LENGTH = 10

# Number of red<->blue (or on<->off) transitions within that history
# required before we call it a genuine flash instead of noise.
FLASH_TRANSITIONS_REQUIRED = 3


def dominant_signal_color(crop_bgr):
    """Returns 'red', 'blue', or None for the dominant siren-like color
    in the top third of a vehicle crop (where a light bar would sit)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    h = crop_bgr.shape[0]
    top = crop_bgr[: max(1, h // 3), :, :]

    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)

    red_mask = (
        ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170))
        & (hsv[:, :, 1] >= 120)
        & (hsv[:, :, 2] >= 120)
    )
    blue_mask = (
        (hsv[:, :, 0] >= 100)
        & (hsv[:, :, 0] <= 130)
        & (hsv[:, :, 1] >= 120)
        & (hsv[:, :, 2] >= 120)
    )

    total_pixels = top.shape[0] * top.shape[1]
    red_fraction = np.count_nonzero(red_mask) / total_pixels
    blue_fraction = np.count_nonzero(blue_mask) / total_pixels

    if red_fraction < COLOR_FRACTION_THRESHOLD and blue_fraction < COLOR_FRACTION_THRESHOLD:
        return None
    return "red" if red_fraction >= blue_fraction else "blue"


class EmergencyDetector:
    """Tracks per-object color history and flags flashing (emergency)
    vehicles. One instance per lane."""

    def __init__(self):
        self.history = {}       # object_id -> deque of last colors (str or None)
        self.flagged_ids = set()

    def update(self, object_id, crop_bgr):
        color = dominant_signal_color(crop_bgr)

        hist = self.history.setdefault(object_id, deque(maxlen=HISTORY_LENGTH))
        hist.append(color)

        if object_id in self.flagged_ids:
            return True

        transitions = 0
        previous = None
        for c in hist:
            if c is not None and previous is not None and c != previous:
                transitions += 1
            if c is not None:
                previous = c

        if transitions >= FLASH_TRANSITIONS_REQUIRED:
            self.flagged_ids.add(object_id)
            return True

        return False

    def any_flagged(self):
        return len(self.flagged_ids) > 0

    def forget(self, object_id):
        self.history.pop(object_id, None)
        self.flagged_ids.discard(object_id)
