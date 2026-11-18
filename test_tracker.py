"""
Tests for tracking/centroidtracker.py - in particular that vehicles are
registered/deregistered correctly when maxDistance rejects a match.
"""

from tracking.centroidtracker import CentroidTracker


def box(cx, cy, half=5):
    return (cx - half, cy - half, cx + half, cy + half)


def test_registers_new_objects():
    ct = CentroidTracker(maxDisappeared=5, maxDistance=50)
    objects = ct.update([box(10, 10), box(200, 200)])
    assert len(objects) == 2


def test_follows_moving_object():
    ct = CentroidTracker(maxDisappeared=5, maxDistance=50)
    ct.update([box(10, 10)])
    objects = ct.update([box(20, 15)])
    assert list(objects.keys()) == [0]
    assert tuple(objects[0]) == (20, 15)


def test_far_new_object_registered_when_tracker_is_not_empty():
    # One tracked car leaves and a new one appears far away in the same
    # frame. The new car must be registered (previously it was silently
    # dropped because #objects >= #inputs).
    ct = CentroidTracker(maxDisappeared=5, maxDistance=50)
    ct.update([box(10, 10)])
    objects = ct.update([box(500, 500)])
    assert ct.nextObjectID == 2
    assert 1 in objects and tuple(objects[1]) == (500, 500)
    assert ct.disappeared[0] == 1


def test_unmatched_object_ages_when_more_inputs_than_objects():
    ct = CentroidTracker(maxDisappeared=1, maxDistance=50)
    ct.update([box(10, 10)])
    ct.update([box(500, 500), box(600, 600)])
    assert ct.disappeared[0] == 1
    objects = ct.update([box(500, 500), box(600, 600)])
    assert 0 not in objects
    assert len(objects) == 2


def test_deregisters_after_max_disappeared():
    ct = CentroidTracker(maxDisappeared=2, maxDistance=50)
    ct.update([box(10, 10)])
    for _ in range(3):
        objects = ct.update([])
    assert len(objects) == 0
