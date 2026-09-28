# Adaptive Traffic Signal Control Using Computer Vision

A computer-vision based traffic signal system that watches a video feed for
each lane of an intersection, detects and counts vehicles in real time, and
computes a fair, congestion-aware green-light duration for every lane.

College project: **P.E.S. College of Engineering, Dept. of Computer Science
and Engineering.** Submitted by Sandesh Bansode (49) and Syed Asimuddin (52).

---

## 1. What it does

For each of the 4 lane videos (`videos/1.mp4` .. `videos/4.mp4`):

1. Runs a **YOLOv4-tiny** object detector on every frame to find vehicles
   (car, bus, truck, motorbike, bicycle).
2. Feeds the detections into a lightweight **centroid tracker** so each
   vehicle is counted exactly once as it crosses the frame, instead of being
   re-counted on every detection.
3. Draws bounding boxes and a running vehicle count directly on the video.
4. All 4 lanes play **simultaneously in one window**, arranged as a 2x2 grid,
   so you can watch every lane detect and count vehicles at the same time.
5. Watches each tracked vehicle for a **flashing red/blue light bar** to flag
   emergency vehicles and preempt the signal for that lane.
6. Once all 4 lanes finish, it feeds the final counts (and any preemption)
   into an **adaptive signal-timing algorithm** (see below) and shows a
   summary screen with the green-light duration computed for each lane.

Two companion scripts extend this:

- **`simulate_comparison.py`** — runs a multi-cycle simulation proving the
  adaptive algorithm actually beats a naive fixed-timer scheme (not just a
  single demo run) and produces a chart.
- **`dashboard.py`** — a Streamlit web dashboard that visualizes the live
  plan, vehicle-count history, backlog, and the fixed-timer-vs-adaptive
  comparison, all from the files the two scripts above write.

## 2. How to run it

```bash
cd "traffic project"
python detect_and_signal.py
```

A window titled **"Adaptive Traffic Signal - Live Detection (4 Lanes)"** will
open showing all 4 lanes at once with live bounding boxes and vehicle counts.

**Controls (click the window first so it has focus):**
| Key | Action |
|-----|--------|
| `q` | Quit / stop early |
| `p` | Pause (press any key to resume) |
| any key | Close the final summary screen once it appears |

When it finishes you'll see console output like:

```
Lane 1: 23 vehicles detected
Lane 2: 41 vehicles detected
Lane 3: 12 vehicles detected
Lane 4: 26 vehicles detected
Saved smooth annotated video to: .../outputs/traffic_detection_output.mp4
Vehicle counts per lane: [23, 41, 12, 26]
Lane 1: demand=23.0 green=46.0s served=23.0 carries_over=0.0
Lane 2: demand=41.0 green=60.0s served=30.0 carries_over=11.0
Lane 3: demand=12.0 green=24.0s served=12.0 carries_over=0.0
Lane 4: demand=26.0 green=52.0s served=26.0 carries_over=0.0
Total cycle time: 182.0s
```
followed by a **"Traffic Signal Summary"** window with a bar chart of the
computed green times.

### Requirements

Already covered if you have Python 3.12+ with:
```
opencv-python
numpy
scipy
matplotlib
streamlit
pandas
eclipse-sumo   # only needed for sumo_sim/ (Module 5) - ~300MB, see below
traci
sumolib
```
No TensorFlow, PyTorch, or `dlib` needed — detection runs entirely through
OpenCV's built-in `cv2.dnn` module, which is why this works on a modern
Python version where the project's original stack does not (see
[§7 Legacy pipeline](#7-legacy-pipeline-not-used) below).

The YOLOv4-tiny model files live in `models/` (`yolov4-tiny.cfg`,
`yolov4-tiny.weights`, `coco.names`) and are already downloaded — nothing
else to fetch.

### Running the extras

```bash
# Prove the algorithm actually helps, over many simulated signal cycles:
python simulate_comparison.py

# Live dashboard (reads the files the two scripts above produce):
streamlit run dashboard.py
```
Run `detect_and_signal.py` (and ideally `simulate_comparison.py`) at least
once before opening the dashboard, otherwise its panels just tell you what's
missing. The dashboard has a manual "Refresh now" button — it doesn't
auto-poll, so re-run a script, then click refresh (or reload the page).

## 3. The adaptive signal algorithm

The old approach sorted counts into fixed buckets (`<25 -> 30s`,
`25-40 -> 60s`, `>40 -> 120s`), which throws away information — a lane with
26 vehicles and one with 39 got treated identically, and a very congested
lane could hog the entire cycle.

Instead, `detect_and_signal.py` uses a **saturation-flow model with
carry-over**, the same idea real adaptive signal systems (SCOOT/SCATS) are
built on:

1. **Saturation rate** — each lane is assumed to clear about 1 vehicle every
   2 seconds of green (`SATURATION_RATE = 0.5` vehicles/sec), a standard
   traffic-engineering estimate for a single lane's discharge rate.
2. **Demand** — a lane's demand is always the *real* number of vehicles: live
   count plus whatever backlog actually carried over from the previous
   cycle. Demand is never artificially inflated — see the pitfall below.
3. **Green time** — `green = clamp(demand / saturation_rate, MIN_GREEN=10s,
   MAX_GREEN=60s)`. The cap means one very congested lane can never
   monopolize the whole intersection; the floor means a near-empty lane still
   gets a minimum green window.
4. **Serve vs. carry-over** — vehicles actually cleared this cycle =
   `green * saturation_rate`. Anything beyond that isn't discarded — it
   becomes next cycle's backlog (`signal_state.json`), so a heavy lane like
   "45 vehicles, only 30 fit in the 60s cap" has its remaining 15 carried
   forward into next cycle's demand.
5. **Aging (anti-starvation)** — a lane that's still backlogged after its
   green phase gets its *cap* raised by `CAP_ESCALATION_STEP = 15s` for every
   consecutive cycle it stays backlogged (up to `ABSOLUTE_MAX_GREEN = 120s`).
   The cap resets back to the 60s baseline the moment the lane fully clears.
   This is what guarantees a heavy lane eventually gets a wide-enough window
   to catch up, without a fixed ceiling starving it forever.

This is what produces results like Lane 2 above: 41 detected, capped at 60s
green, 30 served, **11 carried into the next run's demand** (and next run its
cap would rise to 75s if it's still backlogged).

> **A pitfall worth knowing about (and why it matters for the report):** the
> first version of this algorithm tried to fix starvation by *multiplying*
> the carried-over backlog by a "priority boost" each cycle
> (`demand = count + backlog * 1.5`). That's a bug, not a feature - backlog is
> a real vehicle count, and multiplying it every cycle compounds
> geometrically. `simulate_comparison.py` (see below) caught this
> immediately: after 8 simulated cycles the "adaptive" backlog had exploded
> to 733 vehicles, worse than doing nothing. The fix was to keep demand as
> the real, un-inflated count and instead grow the lane's *green-time cap*
> linearly the longer it stays backlogged (aging) - bounded, convergent, and
> still guarantees priority to a starved lane. This is a good example of why
> the comparison simulation exists: it's easy to design a "fair-sounding"
> rule that is actually unstable, and a multi-cycle simulation catches that
> before a single demo run would.

Tunable constants are all at the top of `detect_and_signal.py`:

| Constant | Meaning |
|---|---|
| `SATURATION_RATE` | vehicles cleared per second of green |
| `MIN_GREEN` / `MAX_GREEN` | floor/baseline cap on green time per lane, seconds |
| `CAP_ESCALATION_STEP` / `ABSOLUTE_MAX_GREEN` | how fast (and how far) a backlogged lane's cap grows per consecutive cycle |
| `EMERGENCY_MIN_GREEN` | minimum green given during an emergency preemption |
| `CONFIDENCE_THRESHOLD` | minimum detector confidence to count a box |
| `DETECT_EVERY_N_FRAMES` | run detection every N frames (tracker fills the gaps) |
| `INPUT_SIZE` | YOLO network input resolution (416 = standard for yolov4-tiny) |

## 4. Emergency vehicle preemption

COCO (what YOLOv4-tiny is trained on) has no "ambulance" or "fire truck"
class - both are just "car"/"truck" to the detector. Instead of a class
label, `emergency.py` looks for what actually distinguishes an emergency
vehicle on camera: a light bar that **flashes** between red and blue. For
every tracked vehicle it samples the dominant color in the top third of its
box each frame; if that color flips between red and blue several times
within a short rolling window, the vehicle is flagged and drawn with a
magenta box + "EMERGENCY" label.

When a lane has a flagged vehicle, `compute_signal_plan` **preempts** it:
that lane's green is extended to clear its *entire* queue in one phase
(bypassing the normal cap), exactly like real signal-preemption hardware
holds green until the priority vehicle clears the intersection.

This is a heuristic, not a trained classifier, and the 4 sample lane videos
don't contain any real emergency vehicles, so you should expect to see it
stay quiet during a normal run - that's correct behavior, not a missing
feature. `test_emergency.py` proves the detection *logic* itself works using
synthetic red/blue flashing crops (no ambulance footage required):
```bash
python test_emergency.py
```

## 5. Before/after: does the algorithm actually help?

A single demo run only shows one snapshot. `simulate_comparison.py` runs 25
simulated signal cycles under two policies - facing the *same* random
vehicle arrivals each time, seeded from the last real detection run
(`out.txt`) - and compares:

- **Fixed-timer (naive):** every lane always gets the same 30s green,
  regardless of demand.
- **Adaptive (this project):** the saturation-flow + aging algorithm above.

```bash
python simulate_comparison.py
```
```
Fixed-timer (naive, 30s every lane): avg queued vehicles/cycle=587.4  final backlog=1117.0  ...
Adaptive (saturation-flow + carry-over): avg queued vehicles/cycle=8.7  final backlog=0.0  ...

Average queue length reduced by 98.5% with the adaptive algorithm.
```
It also saves `outputs/comparison_chart.png` (backlog-over-time for both
schemes) and `outputs/comparison.json` (raw numbers, read by the dashboard).
The fixed-timer scheme's backlog climbs continuously because 30s isn't
enough green to keep up with a busy lane's real arrival rate; the adaptive
scheme's aging mechanism keeps catching it up before it runs away.

## 6. Live dashboard

```bash
streamlit run dashboard.py
```
Opens a browser tab (default `http://localhost:8501`) showing:
- the current signal plan per lane (green time, served, carried-over,
  cycles waited), with an emergency banner if any lane was preempted;
- vehicle count over time per lane, from the run's telemetry
  (`outputs/telemetry.csv`);
- the persisted backlog across lanes (`signal_state.json`);
- the fixed-timer-vs-adaptive comparison chart and metrics, once you've run
  `simulate_comparison.py`.

It only reads files - it doesn't run detection itself, so it's safe to leave
open in a browser tab while `detect_and_signal.py` runs in its own window.
Click **"Refresh now"** after a new run to pull in the latest data.

## 7. SUMO simulation (Module 5)

`simulate_comparison.py` proves the algorithm helps using our own queueing
math. `sumo_sim/` proves it a second, stronger way: it drives an actual
[SUMO](https://sumo.dlr.de) microscopic traffic simulation through TraCI,
using the **exact same** `compute_signal_plan()` from `detect_and_signal.py`
to control a real simulated intersection - an independent traffic simulator
confirming the algorithm actually reduces waiting time for simulated
vehicles, not just in our own model of itself.

```bash
python sumo_sim/generate_routes.py   # (re)builds routes.rou.xml from out.txt's real counts
python sumo_sim/traci_control.py --mode both       # headless: fixed vs. adaptive
python sumo_sim/traci_control.py --mode adaptive --gui   # watch it visually in sumo-gui
```

**What's simulated:** a 4-arm intersection (`sumo_sim/intersection.*.xml`),
one lane per direction, with a **custom 4-phase traffic light** - each lane
gets an exclusive green in turn, mirroring `detect_and_signal.py`'s
independent per-lane model (rather than SUMO's default paired-opposing-
movements program). `generate_routes.py` turns the real per-lane counts from
`out.txt` into traffic volumes, keeping their *relative* proportions (Lane 2
busiest, Lane 3 quietest) rather than the literal count - a 41-vehicle count
from a 10-second clip is a queue snapshot, not a literal "41 vehicles/10s"
arrival rate, so scaling it directly would imply an unrealistic ~14,800
vehicles/hour on one lane.

`traci_control.py` runs two back-to-back simulations that face the identical
traffic demand:
- **fixed:** every lane always gets 30s green, in the same round-robin order.
- **adaptive:** every cycle, it reads the *real* simulated queue length on
  each approach (`traci.edge.getLastStepHaltingNumber`) and feeds it into
  `compute_signal_plan()`, exactly as a live camera feed would.

Both runs continue cycling - past the initial 20 measured cycles if needed -
until (almost) every vehicle has actually completed its trip, not just for a
fixed wall-clock window. This matters: a vehicle still stuck in queue when a
run stops never gets a completion record at all, so cutting a run short
would silently make a *worse* policy look better by dropping its stragglers
from the average.

Latest measured result on the sample data:

```
fixed:    completed_trips=702  avg_waiting_time=131.6s  avg_duration=256.3s  avg_time_loss=166.5s
adaptive: completed_trips=702  avg_waiting_time=53.4s   avg_duration=165.7s  avg_time_loss=75.9s

SUMO-simulated average waiting time reduced by 59.4% with the adaptive algorithm.
```
Same number of completed trips both times (a fair comparison), ~59% less
average waiting time under the adaptive policy. Results are saved to
`outputs/sumo_comparison.json` (read by the dashboard) and
`outputs/sumo_tripinfo_fixed.xml` / `outputs/sumo_tripinfo_adaptive.xml`
(SUMO's raw per-vehicle trip records).

> **Calibration note:** the first attempt at this used unrealistic traffic
> volumes and 200m approach roads, which physically gridlocked two of the
> four lanes regardless of signal policy (their queues maxed out the road's
> physical capacity, ~29 vehicles, under *both* schemes) - not a meaningful
> test of the algorithm. Lengthening the approach roads to 600m and scaling
> volumes to a realistic, moderately-congested level (one lane deliberately
> above the fixed-timer scheme's effective per-lane capacity, to actually
> exercise the aging/escalation logic) is what produced the result above.
> This is exactly the kind of thing a real simulator catches that a demo
> video can't - along with the `STARVATION_BOOST` bug in §3.

## 8. Output files

| File | What it is |
|---|---|
| `out.txt` | Final vehicle count per lane from the last run (one number per line) |
| `signal_state.json` | Persisted backlog + cycles-waited per lane, carried between runs |
| `outputs/traffic_detection_output.mp4` | The full 2x2 grid with boxes + counts, recorded at a fixed 30fps. Live detection speed can be uneven depending on your CPU, so this file is what you should watch/share for guaranteed-smooth playback. |
| `outputs/telemetry.csv` | Per-frame vehicle count / emergency flag per lane over the run, used by the dashboard's history chart |
| `outputs/latest_run.json` | Full signal plan from the last run, used by the dashboard |
| `outputs/comparison_chart.png` / `outputs/comparison.json` | Fixed-timer vs. adaptive simulation results, from `simulate_comparison.py` |
| `outputs/sumo_comparison.json` | Fixed-timer vs. adaptive results from the real SUMO simulation, from `sumo_sim/traci_control.py` |
| `outputs/sumo_tripinfo_fixed.xml` / `outputs/sumo_tripinfo_adaptive.xml` | SUMO's raw per-vehicle trip records for each policy |

## 9. Project structure

```
traffic project/
├── detect_and_signal.py     # main script - run this
├── emergency.py             # flashing-light emergency vehicle heuristic
├── test_emergency.py        # synthetic test for emergency.py (no video needed)
├── simulate_comparison.py   # fixed-timer vs. adaptive, multi-cycle simulation
├── dashboard.py             # Streamlit live dashboard (reads outputs/*, signal_state.json)
├── sumo_sim/                # SUMO microsimulation validation (Module 5)
│   ├── intersection.nod.xml / .edg.xml / .con.xml  # network source (-> netconvert)
│   ├── intersection.net.xml    # built network + custom 4-phase traffic light
│   ├── intersection.sumocfg    # ties network + routes together
│   ├── generate_routes.py      # builds routes.rou.xml from out.txt's real counts
│   ├── routes.rou.xml           # generated - vehicle flows per lane
│   └── traci_control.py         # runs fixed vs. adaptive through TraCI
├── tracking/
│   ├── centroidtracker.py   # assigns/tracks IDs across frames, no dlib needed
│   └── trackableobject.py   # small record: object ID + counted flag
├── models/                  # YOLOv4-tiny weights/config (OpenCV DNN)
│   ├── yolov4-tiny.cfg
│   ├── yolov4-tiny.weights
│   └── coco.names
├── videos/                  # the 4 lane videos + a test clip
├── outputs/                 # generated: annotated video, telemetry, comparison chart/json, latest run
├── out.txt                  # generated vehicle counts (created on first run)
├── signal_state.json        # generated backlog + cycles-waited state (created on first run)
├── fiprjt - 02.docx         # original project proposal document
│
│   --- legacy files below, kept for reference, not used by detect_and_signal.py ---
├── multithreading.py        # old YOLOv3/tensornets/dlib detector (broken on modern Python)
├── program.py                # old fixed-ratio signal-timing script
├── run.sh                    # old shell pipeline (multithreading.py x4 -> program.py)
├── requirements.txt          # old pinned deps (TensorFlow 2.1, dlib 19.19) - do not install
└── requirements-python-3.7.4.txt
```

## 10. Legacy pipeline (not used)

The repo originally shipped with `multithreading.py`, which detected vehicles
using `tensornets` (YOLOv3) on top of `tensorflow==2.1.0`, with `dlib` for
inter-frame tracking. Those exact pinned versions only support Python up to
3.7 and will not install on Python 3.12/3.14, so that pipeline cannot run as
originally written. `detect_and_signal.py` replaces it end-to-end with
OpenCV's DNN module (YOLOv4-tiny) and drops the `dlib` dependency entirely,
while still reusing the original `tracking/centroidtracker.py`. The old files
are left in place for reference but aren't needed to run anything.

## 11. Background: the college project brief

From the project proposal (`fiprjt - 02.docx`):

**Problem statement:** Traditional traffic signals use fixed timings and
can't adapt to changing traffic conditions, causing congestion, long queues,
and unnecessary waiting. This project uses computer vision to analyze
vehicle count, lane density, queue length, and emergency vehicles, and
dynamically adjusts signal timings.

**Planned modules:**

| # | Module | Status |
|---|--------|--------|
| 1 | Vehicle Detection (YOLO) | ✅ implemented (`detect_and_signal.py`) |
| 2 | Traffic Analysis (density / queue length) | ✅ vehicle count used as queue/density proxy |
| 3 | Emergency Vehicle Detection | ✅ implemented (`emergency.py`, flash-color heuristic + preemption) |
| 4 | Signal Optimization | ✅ implemented (saturation-flow + aging algorithm) |
| 5 | Traffic Simulation (SUMO) | ✅ implemented (`sumo_sim/`, TraCI-driven validation) |

**Target users:** Traffic management authorities, smart-city systems,
transportation planners.

**Technology stack:** Python, YOLO, OpenCV, SUMO.
