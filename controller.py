"""
Real-time traffic signal controllers for a single intersection.

Approaches that don't conflict are grouped into PHASES that go green
together - by default the standard two-phase plan: North + South, then
East + West (opposing through traffic doesn't cross). Every phase change
goes through the real-world change intervals:

    GREEN -> YELLOW -> ALL-RED -> (next phase) GREEN -> ...

Both controllers share one interface - call update(dt, queues, ...) every
tick with the current queue on each approach and they return the light
shown on each approach - so the exact same logic drives the camera demo
(detect_and_signal.py), the queue simulation (simulate_comparison.py) and
the SUMO validation (sumo_sim/traci_control.py).

AdaptiveController - how it decides:
  * A phase's queue is that of its busiest approach (the critical lane):
    the green has to be long enough for the longest queue in the phase.
  * Next green goes to the phase with the highest priority:
        queue + AGING_WEIGHT * seconds waited on red
    so long queues go first, but a short queue that has waited a long time
    eventually wins too. Once a phase has waited MAX_RED_WAIT seconds with
    cars queued, the running green is ended (after the minimum green) and
    the longest-waiting phase goes next (starvation guard). An emergency
    vehicle beats everything.
  * Planned green = queue / SATURATION_RATE (time to discharge the critical
    queue), clamped to [min green, max green].
  * The green ends early once every approach in the phase has been empty
    for GAP_TIME seconds and someone else is waiting ("gap-out"), ends at
    the planned time if someone else is waiting ("max-out"), and otherwise
    stays green while nobody else needs it ("rest in green").
  * A phase that maxes out with cars still queued gets its max green
    raised by CAP_ESCALATION_STEP next time (up to ABSOLUTE_MAX_GREEN).
  * An emergency vehicle on a green approach holds the green; one on a red
    approach cuts the current green short (after PREEMPT_MIN_GREEN).

FixedTimeController is the conventional baseline: every phase gets
FIXED_GREEN seconds in a fixed rotation, whatever the traffic.
"""

import math

GREEN = "green"
YELLOW = "yellow"
RED = "red"
ALL_RED = "all_red"

SATURATION_RATE = 0.5      # vehicles an approach discharges per second of green
MIN_GREEN = 10             # seconds
MAX_GREEN = 60             # seconds - baseline cap per green phase
CAP_ESCALATION_STEP = 15   # extra cap per consecutive max-out
ABSOLUTE_MAX_GREEN = 120   # hard ceiling, including escalation and emergency holds
GAP_THRESHOLD = 0          # queue at or below this counts as "empty"
GAP_TIME = 2.0             # seconds a phase must stay empty before gapping out
AGING_WEIGHT = 0.1         # priority per second waited (10 s of waiting ~ 1 queued car)
MAX_RED_WAIT = 120         # seconds - starvation guard
PREEMPT_MIN_GREEN = 5      # shortest green allowed before an emergency preempts it

FIXED_GREEN = 30

# --- Real-world change intervals (ITE kinematic formulas) --------------------

PERCEPTION_REACTION_S = 1.0   # driver perception-reaction time
DECELERATION = 3.05           # m/s^2, comfortable deceleration (10 ft/s^2)
GRAVITY = 9.81
VEHICLE_LENGTH_M = 5.0
PED_WALK_S = 7.0              # pedestrian WALK interval
PED_SPEED = 1.2               # m/s, design walking speed

DEFAULT_SPEED_KMH = 50
DEFAULT_INTERSECTION_WIDTH_M = 20


def _round_up_half(seconds):
    return math.ceil(seconds * 2) / 2


def change_intervals(speed_kmh=DEFAULT_SPEED_KMH, intersection_width_m=DEFAULT_INTERSECTION_WIDTH_M,
                     grade=0.0):
    """Yellow and all-red times for an approach speed and intersection width.

    yellow  = t + v / (2a + 2Gg)  - long enough to either stop comfortably
                                    or reach the stop line (no dilemma zone)
    all-red = (W + L) / v         - time for a car that entered on the last
                                    instant of yellow to clear the junction
    Rounded up to 0.5 s; yellow kept within the usual 3-6 s range.
    """
    v = speed_kmh / 3.6
    yellow = PERCEPTION_REACTION_S + v / (2 * DECELERATION + 2 * GRAVITY * grade)
    all_red = (intersection_width_m + VEHICLE_LENGTH_M) / v
    return min(6.0, max(3.0, _round_up_half(yellow))), max(1.0, _round_up_half(all_red))


def pedestrian_min_green(crossing_width_m=DEFAULT_INTERSECTION_WIDTH_M):
    """Shortest green that lets a pedestrian who steps off at the start of
    WALK finish crossing in parallel with the vehicle green."""
    return _round_up_half(PED_WALK_S + crossing_width_m / PED_SPEED)


YELLOW_TIME, ALL_RED_TIME = change_intervals()


def two_phase_plan(num_lanes=4):
    """Standard 4-way plan: approaches 0+2 (North+South), then 1+3 (East+West)."""
    if num_lanes != 4:
        return single_phase_plan(num_lanes)
    return [[0, 2], [1, 3]]


def single_phase_plan(num_lanes):
    """Each approach on its own (split phasing / protected turns)."""
    return [[i] for i in range(num_lanes)]


class _BaseController:
    def __init__(self, num_lanes, phases=None, yellow_time=YELLOW_TIME, all_red_time=ALL_RED_TIME):
        self.num_lanes = num_lanes
        self.phases = [list(p) for p in (phases or single_phase_plan(num_lanes))]
        seen = [lane for phase in self.phases for lane in phase]
        if sorted(seen) != list(range(num_lanes)):
            raise ValueError("every approach must be in exactly one phase")
        self.phase_of = {lane: p for p, phase in enumerate(self.phases) for lane in phase}
        num_phases = len(self.phases)

        self.yellow_time = yellow_time
        self.all_red_time = all_red_time

        self.time = 0.0
        self.phase = ALL_RED
        self.green_phase = None         # phase holding green or yellow
        # Start "already done" with all-red so the first update picks a phase
        self.phase_elapsed = all_red_time
        self.planned_green = 0.0
        self.last_served = None
        self.last_green_start = [None] * num_phases
        self.red_wait = [0.0] * num_phases
        self.phase_log = []
        self.choice = None              # why the current/last green phase was picked

    @property
    def num_phases(self):
        return len(self.phases)

    @property
    def green_lanes(self):
        return list(self.phases[self.green_phase]) if self.green_phase is not None else []

    def lane_red_wait(self, lane):
        return self.red_wait[self.phase_of[lane]]

    def lights(self):
        lights = [RED] * self.num_lanes
        for lane in self.green_lanes:
            lights[lane] = GREEN if self.phase == GREEN else YELLOW
        return lights

    def time_remaining(self):
        """Seconds left in the current interval (planned, for GREEN)."""
        if self.phase == GREEN:
            return max(0.0, self.planned_green - self.phase_elapsed)
        if self.phase == YELLOW:
            return max(0.0, self.yellow_time - self.phase_elapsed)
        return max(0.0, self.all_red_time - self.phase_elapsed)

    def _phase_inputs(self, queues, emergency, active):
        """Per-approach inputs -> per-phase (queue, emergency, active)."""
        pq, pe, pa = [], [], []
        for phase in self.phases:
            live = [lane for lane in phase if active[lane]]
            pq.append(max((queues[lane] for lane in live), default=0))
            pe.append(any(emergency[lane] for lane in live))
            pa.append(bool(live))
        return pq, pe, pa

    def update(self, dt, queues, emergency=None, active=None):
        if emergency is None:
            emergency = [False] * self.num_lanes
        if active is None:
            active = [True] * self.num_lanes
        pq, pe, pa = self._phase_inputs(queues, emergency, active)

        self.time += dt
        self.phase_elapsed += dt
        for p in range(self.num_phases):
            if p != self.green_phase and pa[p] and pq[p] > GAP_THRESHOLD:
                self.red_wait[p] += dt

        if self.phase == GREEN:
            reason = self._green_end_reason(dt, pq, pe, pa)
            if reason is not None:
                entry = self.phase_log[-1]
                entry["green"] = round(self.phase_elapsed, 2)
                entry["end_reason"] = reason
                entry["queue_at_end"] = pq[self.green_phase]
                self._on_green_end(reason)
                self.phase = YELLOW
                self.phase_elapsed = 0.0
        elif self.phase == YELLOW:
            if self.phase_elapsed >= self.yellow_time:
                self.last_served = self.green_phase
                self.green_phase = None
                self.phase = ALL_RED
                self.phase_elapsed = 0.0
        elif self.phase == ALL_RED:
            if self.phase_elapsed >= self.all_red_time:
                p = self._choose_next(pq, pe, pa)
                if p is not None:
                    self._start_green(p, pq[p], pe[p], queues)

        return self.lights()

    def _start_green(self, p, queue, is_emergency, lane_queues):
        self.planned_green = self._planned_green(p, queue)
        self.phase_log.append({
            "phase": p,
            "lanes": list(self.phases[p]),
            "start": round(self.time, 2),
            "queue_at_start": queue,
            "lane_queues": [lane_queues[lane] for lane in self.phases[p]],
            "waited": round(self.red_wait[p], 2),
            "planned": round(self.planned_green, 2),
            "green": None,
            "end_reason": None,
            "choice": self.choice,
            "emergency": bool(is_emergency),
        })
        self.green_phase = p
        self.phase = GREEN
        self.phase_elapsed = 0.0
        self.red_wait[p] = 0.0
        self.last_green_start[p] = self.time

    def _longest_unserved(self, phases):
        # Never-served phases first, then the one whose last green is oldest
        return min(phases, key=lambda p: (self.last_green_start[p] is not None,
                                          self.last_green_start[p] or 0.0, p))

    def _on_green_end(self, reason):
        pass


class AdaptiveController(_BaseController):
    def __init__(self, num_lanes, phases=None, min_green=MIN_GREEN, max_green=MAX_GREEN,
                 saturation_rate=SATURATION_RATE, yellow_time=YELLOW_TIME,
                 all_red_time=ALL_RED_TIME, gap_time=GAP_TIME, max_red_wait=MAX_RED_WAIT):
        super().__init__(num_lanes, phases, yellow_time, all_red_time)
        self.min_green = min_green
        self.max_green = max(max_green, min_green)
        self.saturation_rate = saturation_rate
        self.gap_time = gap_time
        self.max_red_wait = max_red_wait
        self.maxout_streak = [0] * self.num_phases
        self.empty_for = 0.0

    def priority(self, p, queue):
        return queue + AGING_WEIGHT * self.red_wait[p]

    def effective_max_green(self, p):
        return min(ABSOLUTE_MAX_GREEN, self.max_green + self.maxout_streak[p] * CAP_ESCALATION_STEP)

    def _planned_green(self, p, queue):
        ideal = queue / self.saturation_rate
        return max(self.min_green, min(self.effective_max_green(p), ideal))

    def _choose_next(self, pq, pe, pa):
        candidates = [p for p in range(self.num_phases) if pa[p]]
        if not candidates:
            return None

        urgent = [p for p in candidates if pe[p]]
        if urgent:
            self.choice = "emergency"
            return max(urgent, key=lambda p: pq[p])

        waiting = [p for p in candidates if pq[p] > GAP_THRESHOLD]
        # Don't hand the green straight back to the phase that just had it
        # while somebody else is queued.
        if len(waiting) > 1 and self.last_served in waiting:
            waiting.remove(self.last_served)

        if waiting:
            starving = [p for p in waiting if self.red_wait[p] >= self.max_red_wait]
            if starving:
                self.choice = "max-wait"
                return max(starving, key=lambda p: self.red_wait[p])
            self.choice = "priority"
            return max(waiting, key=lambda p: (self.priority(p, pq[p]), -p))

        # Nobody is queued: rotate through the phases so every approach keeps
        # getting looked at (and, in the video demo, its footage advances).
        if len(candidates) > 1 and self.last_served in candidates:
            candidates.remove(self.last_served)
        self.choice = "idle rotation"
        return self._longest_unserved(candidates)

    def _green_end_reason(self, dt, pq, pe, pa):
        p = self.green_phase
        elapsed = self.phase_elapsed

        if pq[p] <= GAP_THRESHOLD:
            self.empty_for += dt
        else:
            self.empty_for = 0.0

        if not pa[p]:
            return "cleared"
        if elapsed >= ABSOLUTE_MAX_GREEN:
            return "max-out"

        others = [o for o in range(self.num_phases) if o != p]
        if any(pe[o] and pa[o] for o in others) and not pe[p] and elapsed >= PREEMPT_MIN_GREEN:
            return "preempted"
        if pe[p]:
            return None
        if elapsed < self.min_green:
            return None

        others_waiting = any(pa[o] and pq[o] > GAP_THRESHOLD for o in others)
        others_starving = any(pa[o] and pq[o] > GAP_THRESHOLD and self.red_wait[o] >= self.max_red_wait
                              for o in others)
        if others_starving:
            return "max-wait"
        if self.empty_for >= self.gap_time and others_waiting:
            return "gap-out"
        if elapsed >= self.planned_green and others_waiting:
            return "max-out"
        if elapsed >= self.planned_green and not others_waiting and self.empty_for >= self.gap_time:
            # Empty and nobody else is waiting: let the rotation look at the
            # other phases rather than resting here indefinitely.
            if any(pa[o] for o in others):
                return "gap-out"
        return None

    def _on_green_end(self, reason):
        p = self.green_phase
        self.empty_for = 0.0
        if reason == "max-out":
            self.maxout_streak[p] += 1
        elif reason != "max-wait":
            self.maxout_streak[p] = 0


class FixedTimeController(_BaseController):
    """Conventional pre-timed signal: same green for every phase in a fixed
    rotation, regardless of traffic (empty phases still get their full green)."""

    def __init__(self, num_lanes, phases=None, green_time=FIXED_GREEN, yellow_time=YELLOW_TIME,
                 all_red_time=ALL_RED_TIME):
        super().__init__(num_lanes, phases, yellow_time, all_red_time)
        self.green_time = green_time

    def _planned_green(self, p, queue):
        return self.green_time

    def _choose_next(self, pq, pe, pa):
        if not any(pa):
            return None
        self.choice = "rotation"
        return 0 if self.last_served is None else (self.last_served + 1) % self.num_phases

    def _green_end_reason(self, dt, pq, pe, pa):
        if self.phase_elapsed >= self.green_time:
            return "timer"
        return None
