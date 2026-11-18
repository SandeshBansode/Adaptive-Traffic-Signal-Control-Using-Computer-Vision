"""
Tests for controller.py - phase sequencing, lane selection, green timing,
gap-out / max-out, starvation guard and emergency preemption.
"""

import pytest

from controller import (
    ABSOLUTE_MAX_GREEN,
    ALL_RED,
    CAP_ESCALATION_STEP,
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
    two_phase_plan,
)

DT = 0.5


def run(ctrl, seconds, queues, emergency=None, active=None):
    """Advances the controller with constant inputs; returns the lights seen each tick."""
    history = []
    for _ in range(int(round(seconds / DT))):
        history.append(ctrl.update(DT, queues, emergency, active))
    return history


def test_first_green_goes_to_longest_queue():
    ctrl = AdaptiveController(4)
    lights = ctrl.update(DT, [3, 12, 5, 0])
    assert lights == [RED, GREEN, RED, RED]
    assert ctrl.planned_green == pytest.approx(12 / SATURATION_RATE)


def test_only_one_approach_is_ever_green():
    ctrl = AdaptiveController(4)
    for lights in run(ctrl, 600, [8, 8, 8, 8]):
        assert sum(light != RED for light in lights) <= 1


def test_green_yellow_allred_sequence():
    ctrl = AdaptiveController(2, min_green=10)
    seen = []
    for _ in range(80):
        ctrl.update(DT, [10, 10])
        state = (ctrl.phase, ctrl.green_phase)
        if not seen or seen[-1] != state:
            seen.append(state)
    assert seen[:4] == [(GREEN, 0), (YELLOW, 0), (ALL_RED, None), (GREEN, 1)]


def test_min_green_is_respected_even_if_lane_empties():
    ctrl = AdaptiveController(2)
    ctrl.update(DT, [10, 5])
    assert ctrl.green_phase == 0
    run(ctrl, MIN_GREEN - 1, [0, 5])
    assert ctrl.phase == GREEN and ctrl.green_phase == 0


def test_gap_out_when_lane_empties_and_others_wait():
    ctrl = AdaptiveController(2)
    ctrl.update(DT, [40, 5])
    run(ctrl, MIN_GREEN + 3, [0, 5])
    assert ctrl.phase_log[0]["end_reason"] == "gap-out"
    assert ctrl.phase_log[0]["green"] < 40 / SATURATION_RATE


def test_max_out_at_planned_green_when_others_wait():
    ctrl = AdaptiveController(2)
    ctrl.update(DT, [20, 5])
    run(ctrl, 45, [20, 5])
    entry = ctrl.phase_log[0]
    assert entry["end_reason"] == "max-out"
    assert entry["green"] == pytest.approx(20 / SATURATION_RATE, abs=DT)


def test_green_is_capped_at_max_green():
    ctrl = AdaptiveController(2)
    ctrl.update(DT, [500, 5])
    assert ctrl.planned_green == MAX_GREEN


def test_rests_in_green_when_nobody_else_waits():
    ctrl = AdaptiveController(2)
    ctrl.update(DT, [5, 0])
    run(ctrl, 3 * MAX_GREEN, [5, 0])
    assert ctrl.phase == GREEN and ctrl.green_phase == 0


def test_does_not_reserve_same_lane_while_others_wait():
    ctrl = AdaptiveController(2)
    ctrl.update(DT, [100, 2])
    run(ctrl, MAX_GREEN + 10, [100, 2])
    assert ctrl.phase_log[1]["phase"] == 1


def test_cap_escalates_after_repeated_max_out():
    ctrl = AdaptiveController(2)
    ctrl.update(DT, [500, 50])
    run(ctrl, 400, [500, 50])
    lane0 = [e for e in ctrl.phase_log if e["phase"] == 0]
    assert lane0[0]["planned"] == MAX_GREEN
    assert lane0[1]["planned"] == MAX_GREEN + CAP_ESCALATION_STEP
    assert all(e["planned"] <= ABSOLUTE_MAX_GREEN for e in ctrl.phase_log)


def test_aging_eventually_serves_small_queue():
    # Lanes 0 and 1 are always heavier than lane 2, but lane 2 must still
    # get a green within a bounded time.
    ctrl = AdaptiveController(3)
    run(ctrl, 600, [30, 30, 1])
    assert any(e["phase"] == 2 for e in ctrl.phase_log)


def test_starvation_guard():
    # Two saturated approaches with escalating caps must not starve the
    # light one: its wait stays bounded by the limit plus at most one
    # minimum green (and its change interval) from each other approach.
    ctrl = AdaptiveController(3, max_red_wait=60)
    run(ctrl, 1800, [200, 200, 1])
    lane2 = [e for e in ctrl.phase_log if e["phase"] == 2]
    bound = 60 + 2 * (MIN_GREEN + 3 + 1) + 1
    assert len(lane2) >= 5
    assert all(e["waited"] <= bound for e in lane2)
    assert any(e["end_reason"] == "max-wait" for e in ctrl.phase_log)


def test_idle_rotation_when_no_demand():
    ctrl = AdaptiveController(3)
    run(ctrl, 120, [0, 0, 0])
    assert {e["phase"] for e in ctrl.phase_log} == {0, 1, 2}


def test_inactive_lanes_are_skipped_and_cleared():
    ctrl = AdaptiveController(3)
    ctrl.update(DT, [10, 20, 5], active=[True, False, True])
    assert ctrl.green_phase == 0
    ctrl.update(DT, [10, 20, 5], active=[False, False, True])
    assert ctrl.phase == YELLOW
    assert ctrl.phase_log[0]["end_reason"] == "cleared"


def test_returns_none_when_everything_finished():
    ctrl = AdaptiveController(2)
    lights = ctrl.update(DT, [0, 0], active=[False, False])
    assert lights == [RED, RED] and ctrl.green_phase is None


def test_emergency_preempts_other_green():
    ctrl = AdaptiveController(2)
    ctrl.update(DT, [50, 2])
    run(ctrl, 6, [50, 2], emergency=[False, True])
    assert ctrl.phase_log[0]["end_reason"] == "preempted"
    run(ctrl, 5, [50, 2], emergency=[False, True])
    assert ctrl.green_phase == 1 and ctrl.phase_log[1]["emergency"]


def test_emergency_holds_green():
    ctrl = AdaptiveController(2)
    ctrl.update(DT, [4, 30], emergency=[True, False])
    run(ctrl, 40, [0, 30], emergency=[True, False])
    assert ctrl.phase == GREEN and ctrl.green_phase == 0


def test_fixed_timer_rotates_with_constant_green():
    ctrl = FixedTimeController(3, green_time=20)
    run(ctrl, 200, [0, 50, 1])
    lanes = [e["phase"] for e in ctrl.phase_log]
    assert lanes[:4] == [0, 1, 2, 0]
    assert all(e["green"] == pytest.approx(20, abs=DT) for e in ctrl.phase_log if e["green"])


def test_time_remaining_counts_down():
    ctrl = AdaptiveController(2)
    ctrl.update(DT, [10, 5])
    before = ctrl.time_remaining()
    ctrl.update(DT, [10, 5])
    assert ctrl.time_remaining() == pytest.approx(before - DT)


# --- Phases: non-conflicting approaches share the green ----------------------

def test_two_phase_opposing_approaches_go_green_together():
    ctrl = AdaptiveController(4, phases=two_phase_plan(4))
    lights = ctrl.update(DT, [3, 12, 5, 0])
    assert lights == [RED, GREEN, RED, GREEN]          # East + West
    assert ctrl.green_lanes == [1, 3]


def test_conflicting_phases_never_green_at_once():
    ctrl = AdaptiveController(4, phases=two_phase_plan(4))
    for _ in range(2400):
        lights = ctrl.update(DT, [8, 15, 2, 6])
        north_south = lights[0] != RED or lights[2] != RED
        east_west = lights[1] != RED or lights[3] != RED
        assert not (north_south and east_west)


def test_phase_green_sized_for_busiest_approach():
    ctrl = AdaptiveController(4, phases=two_phase_plan(4))
    ctrl.update(DT, [4, 0, 20, 0])
    assert ctrl.green_phase == 0
    assert ctrl.planned_green == pytest.approx(20 / SATURATION_RATE)
    assert ctrl.phase_log[0]["lane_queues"] == [4, 20]


def test_phase_gaps_out_only_when_all_its_approaches_are_empty():
    ctrl = AdaptiveController(4, phases=two_phase_plan(4))
    ctrl.update(DT, [30, 5, 30, 5])
    run(ctrl, MIN_GREEN + 5, [0, 5, 30, 5])     # North empty, South still queued
    assert ctrl.phase == GREEN and ctrl.green_phase == 0
    run(ctrl, 4, [0, 5, 0, 5])                  # now both empty
    assert ctrl.phase_log[0]["end_reason"] == "gap-out"


def test_phase_stays_active_while_any_approach_has_video():
    ctrl = AdaptiveController(4, phases=two_phase_plan(4))
    ctrl.update(DT, [5, 0, 5, 0])
    run(ctrl, 3, [0, 0, 5, 0], active=[False, True, True, True])
    assert ctrl.green_phase == 0 and ctrl.phase == GREEN


def test_emergency_on_one_approach_greens_its_whole_phase():
    ctrl = AdaptiveController(4, phases=two_phase_plan(4))
    ctrl.update(DT, [50, 2, 50, 2])
    run(ctrl, 12, [50, 2, 50, 2], emergency=[False, True, False, False])
    assert ctrl.green_lanes == [1, 3]
    assert ctrl.phase_log[-1]["emergency"]


def test_every_approach_needs_exactly_one_phase():
    with pytest.raises(ValueError):
        AdaptiveController(4, phases=[[0, 2], [1]])
    with pytest.raises(ValueError):
        AdaptiveController(4, phases=[[0, 1, 2], [1, 3]])


# --- Real-world timing rules -------------------------------------------------

def test_change_intervals_follow_ite_formula():
    # 50 km/h, 20 m wide: yellow = 1 + 13.9 / 6.1 = 3.3 -> 3.5 s,
    # all-red = (20 + 5) / 13.9 = 1.8 -> 2.0 s
    assert change_intervals(50, 20) == (3.5, 2.0)
    yellow_fast, all_red_fast = change_intervals(70, 20)
    assert yellow_fast > 3.5 and all_red_fast <= 2.0
    assert change_intervals(50, 40)[1] > 2.0          # wider junction needs more all-red
    assert change_intervals(20, 10)[0] == 3.0         # never below 3 s of yellow


def test_pedestrian_min_green_covers_walk_and_crossing():
    assert pedestrian_min_green(12) == pytest.approx(17.0)   # 7 s walk + 12 m / 1.2 m/s


def test_controller_uses_configured_change_intervals():
    ctrl = AdaptiveController(2, yellow_time=4.0, all_red_time=2.0)
    ctrl.update(DT, [10, 10])
    run(ctrl, 20, [10, 10])                     # planned 20 s -> max-out
    assert ctrl.phase == YELLOW
    run(ctrl, 4, [10, 10])
    assert ctrl.phase == ALL_RED
    run(ctrl, 2, [10, 10])
    assert ctrl.phase == GREEN and ctrl.green_phase == 1
