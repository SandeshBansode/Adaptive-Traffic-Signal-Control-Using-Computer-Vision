"""
Tests for the adaptive signal-timing model in detect_and_signal.py:
saturation-flow green times, carry-over backlog, aging/cap escalation,
emergency preemption and the persisted signal state.
"""

import json

import pytest

import detect_and_signal as ds
from detect_and_signal import (
    ABSOLUTE_MAX_GREEN,
    CAP_ESCALATION_STEP,
    EMERGENCY_MIN_GREEN,
    MAX_GREEN,
    MIN_GREEN,
    SATURATION_RATE,
    compute_signal_plan,
)


def test_green_proportional_to_demand():
    plan, backlog, waited = compute_signal_plan([10], [0.0])
    assert plan[0]["green"] == pytest.approx(10 / SATURATION_RATE)
    assert plan[0]["served"] == 10
    assert backlog == [0.0]
    assert waited == [0]


def test_empty_lane_still_gets_min_green():
    plan, backlog, _ = compute_signal_plan([0], [0.0])
    assert plan[0]["green"] == MIN_GREEN
    assert backlog == [0.0]


def test_light_demand_clamped_to_min_green():
    plan, _, _ = compute_signal_plan([1], [0.0])
    assert plan[0]["green"] == MIN_GREEN


def test_heavy_lane_capped_and_carries_over():
    count = 100
    plan, backlog, waited = compute_signal_plan([count], [0.0])
    capacity = MAX_GREEN * SATURATION_RATE
    assert plan[0]["green"] == MAX_GREEN
    assert plan[0]["served"] == capacity
    assert backlog == [pytest.approx(count - capacity)]
    assert waited == [1]


def test_backlog_adds_to_demand():
    plan, _, _ = compute_signal_plan([5], [7.0])
    assert plan[0]["demand"] == 12
    assert plan[0]["backlog_in"] == 7.0


def test_cap_escalates_with_cycles_waited():
    plan, _, _ = compute_signal_plan([200], [0.0], cycles_waited=[2])
    assert plan[0]["green"] == MAX_GREEN + 2 * CAP_ESCALATION_STEP


def test_cap_never_exceeds_absolute_max():
    plan, _, _ = compute_signal_plan([1000], [0.0], cycles_waited=[100])
    assert plan[0]["green"] == ABSOLUTE_MAX_GREEN


def test_persistent_overload_converges():
    # A lane that is over capacity once (but whose arrivals are otherwise
    # sustainable) must eventually clear thanks to aging.
    backlog, waited = [0.0], [0]
    counts = [150] + [20] * 10
    for c in counts:
        _, backlog, waited = compute_signal_plan([c], backlog, waited)
    assert backlog == [0.0]
    assert waited == [0]


def test_emergency_clears_entire_queue():
    plan, backlog, waited = compute_signal_plan([100], [50.0], [3], emergency_flags=[True])
    assert plan[0]["emergency"] is True
    assert plan[0]["served"] == 150
    assert plan[0]["green"] == pytest.approx(150 / SATURATION_RATE)
    assert backlog == [0.0]
    assert waited == [0]


def test_emergency_with_no_demand_uses_emergency_min_green():
    plan, _, _ = compute_signal_plan([0], [0.0], emergency_flags=[True])
    assert plan[0]["green"] == EMERGENCY_MIN_GREEN


def test_lanes_are_independent():
    plan, backlog, waited = compute_signal_plan([5, 100, 0, 20], [0.0] * 4)
    assert [p["green"] for p in plan] == [MIN_GREEN, MAX_GREEN, MIN_GREEN, 40]
    assert backlog[1] > 0 and backlog[0] == backlog[2] == backlog[3] == 0.0
    assert waited == [0, 1, 0, 0]


def test_served_plus_leftover_equals_demand():
    for count, waiting in [(0, 0.0), (3, 1.5), (30, 0.0), (77, 12.0), (500, 40.0)]:
        plan, _, _ = compute_signal_plan([count], [waiting])
        p = plan[0]
        assert p["served"] + p["leftover"] == pytest.approx(p["demand"], abs=0.11)


def test_signal_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "STATE_PATH", str(tmp_path / "state.json"))
    ds.save_signal_state([1.5, 0.0, 2.0, 0.0], [1, 0, 2, 0])
    assert ds.load_signal_state(4) == ([1.5, 0.0, 2.0, 0.0], [1, 0, 2, 0])


def test_signal_state_defaults_when_missing_or_mismatched(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    monkeypatch.setattr(ds, "STATE_PATH", str(path))
    assert ds.load_signal_state(4) == ([0.0] * 4, [0] * 4)

    path.write_text(json.dumps({"backlog": [1.0, 2.0], "cycles_waited": [0, 0]}))
    assert ds.load_signal_state(4) == ([0.0] * 4, [0] * 4)

    path.write_text("not json")
    assert ds.load_signal_state(4) == ([0.0] * 4, [0] * 4)
