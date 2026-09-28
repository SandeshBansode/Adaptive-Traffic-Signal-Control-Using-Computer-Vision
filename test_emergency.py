"""
Synthetic test for the emergency-vehicle flash heuristic in emergency.py.
No video/ambulance footage needed - we build fake crops with a known
red/blue alternating light bar and check the detector reacts correctly,
and that a plain (non-flashing) vehicle never gets flagged.
"""

import numpy as np

from emergency import EmergencyDetector, dominant_signal_color


def make_crop(top_color):
    crop = np.full((60, 60, 3), (40, 40, 40), dtype=np.uint8)  # dark car body
    if top_color == "red":
        crop[0:20, :, :] = (0, 0, 255)   # BGR red
    elif top_color == "blue":
        crop[0:20, :, :] = (255, 0, 0)   # BGR blue
    return crop


def test_dominant_signal_color():
    assert dominant_signal_color(make_crop("red")) == "red"
    assert dominant_signal_color(make_crop("blue")) == "blue"
    assert dominant_signal_color(make_crop(None)) is None


def test_flashing_vehicle_gets_flagged():
    detector = EmergencyDetector()
    sequence = ["red", "blue", "red", "blue", "red", "blue"]

    flagged_at = None
    for i, color in enumerate(sequence):
        if detector.update(object_id=1, crop_bgr=make_crop(color)):
            flagged_at = i
            break

    assert flagged_at is not None, "flashing vehicle was never flagged"
    assert 1 in detector.flagged_ids


def test_plain_vehicle_never_flagged():
    detector = EmergencyDetector()
    for _ in range(10):
        detector.update(object_id=2, crop_bgr=make_crop(None))

    assert 2 not in detector.flagged_ids
    assert not detector.any_flagged()


def test_solid_red_never_flagged():
    # A solid red car (no alternation) should NOT be treated as an
    # emergency vehicle - only flashing/alternating counts.
    detector = EmergencyDetector()
    for _ in range(10):
        detector.update(object_id=3, crop_bgr=make_crop("red"))

    assert 3 not in detector.flagged_ids


if __name__ == "__main__":
    test_dominant_signal_color()
    test_flashing_vehicle_gets_flagged()
    test_plain_vehicle_never_flagged()
    test_solid_red_never_flagged()
    print("All emergency-detector tests passed.")
