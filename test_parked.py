"""
Tests for the parked-vehicle filter and the "count only moving vehicles"
rule in detect_and_signal.LaneCamera.
"""

from detect_and_signal import MOVE_THRESHOLD_PX, PARKED_AFTER_S, LaneCamera
from tracking.trackableobject import TrackableObject

FPS = 30.0


def make_camera():
    cam = LaneCamera("does-not-exist.mp4", "North", roi=None)
    cam.fps = FPS
    return cam


def observe(cam, to, centroid, frame_idx):
    """Simulates the tracker matching `to` at `centroid` on `frame_idx`."""
    cam.frame_idx = frame_idx
    to.centroids.append(centroid)
    to.frames.append(frame_idx)
    cam._update_motion(to, PARKED_AFTER_S * FPS)


def new_object(start=(100, 100), frame_idx=1):
    to = TrackableObject(1, start)
    to.frames.append(frame_idx)
    return to


def test_moving_vehicle_is_counted_once_and_never_parked():
    cam, to = make_camera(), new_object()
    for f in range(2, 900, 2):
        observe(cam, to, (100, 100 + f // 4), f)
    assert cam.counted == 1
    assert not to.parked


def test_stationary_vehicle_is_parked_after_window_and_never_counted():
    cam, to = make_camera(), new_object()
    window = int(PARKED_AFTER_S * FPS)
    observe(cam, to, (102, 99), window - 10)          # jitter only, window not full yet
    assert not to.parked
    observe(cam, to, (99, 101), window + 10)
    assert to.parked
    assert cam.counted == 0


def test_detection_jitter_is_not_movement():
    cam, to = make_camera(), new_object()
    jitter = MOVE_THRESHOLD_PX // 2
    for i, f in enumerate(range(2, int(PARKED_AFTER_S * FPS) + 60, 2)):
        observe(cam, to, (100 + (jitter if i % 2 else -jitter), 100), f)
    assert to.parked and cam.counted == 0


def test_queued_car_that_moves_on_green_is_not_parked():
    # Stands still for a while (start of green, waiting for the queue ahead),
    # then drives off before the parked window elapses.
    cam, to = make_camera(), new_object()
    for f in range(2, int(PARKED_AFTER_S * FPS) - 30, 2):
        observe(cam, to, (100, 100), f)
    observe(cam, to, (100, 140), int(PARKED_AFTER_S * FPS))
    observe(cam, to, (100, 180), int(PARKED_AFTER_S * FPS) + 30)
    assert not to.parked
    assert cam.counted == 1


def test_parked_car_that_drives_away_is_released():
    cam, to = make_camera(), new_object()
    window = int(PARKED_AFTER_S * FPS)
    observe(cam, to, (100, 100), window + 30)
    assert to.parked
    observe(cam, to, (100, 160), window + 60)
    assert not to.parked
    assert cam.counted == 1
