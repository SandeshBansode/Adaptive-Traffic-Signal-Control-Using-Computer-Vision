<div align="center">

# 🚦 Adaptive Traffic Signal Control
### Computer Vision • Intelligent Signal Timing • Traffic Simulation

**A camera-actuated traffic signal: it counts the vehicles on each road, decides which approach gets green and for how long, and pauses the traffic on red approaches, like a real adaptive intersection.**

<br>

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLOv4--tiny-00FFFF?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![SUMO](https://img.shields.io/badge/SUMO-TraCI-222222?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/diyorbek57/Adaptive-Traffic-Signal-Control-Using-Computer-Vision/blob/main/colab_demo.ipynb)

<br><br>

**P.E.S. College of Engineering · Computer Science & Engineering**

**Sandesh Bansode**

</div>

---

## ✦ Overview

Traditional fixed-time signals give every approach the same green whether
it has 30 cars waiting or none. This project replaces that with a
**camera-actuated adaptive controller** that works like a real one:

- Each of the four videos is one approach to an intersection: North, East, South and West
- **YOLOv4-tiny** detects vehicles. Only vehicles **between the two
  road-edge lines** drawn around that camera's inbound lanes are counted,
  and cars that sit still while their traffic has green are treated as
  **parked** and ignored
- The vehicles in that road zone are the approach's **queue**
- The controller follows **real-world signal rules**:
  - Opposing approaches share the green: **North + South**, then **East + West**
  - Every change goes through a **yellow** and an **all-red** interval, calculated from the speed limit and the junction width
- **Red approach → its video is paused** (traffic is stopped).
  **Green approach → its video plays** (traffic moves)
- The controller keeps re-reading the queues to decide **when to switch**
  and **who goes next**
- The same controller is validated in a queue simulation and in **SUMO**

---

## ◉ What You See

The live window shows all four approaches in a 2 × 2 grid. Each tile has:

- A **signal head** and a light badge, for example `GREEN 12s`, `RED waited 23s` or `CLEARED`
- The **road zone**: two yellow edge lines with the shaded area between them
- **Green boxes** for counted vehicles (with tracking IDs)
- **Grey boxes** for vehicles outside the lines, such as the opposite carriageway, pavements and side roads
- **Dashed `P parked` boxes** for vehicles inside the lines that don't move with traffic
- The **queue** (vehicles in the zone now) and the running **count**
- **Dimmed, paused footage** while the approach is red

A status bar shows the signal clock, the current phase, and **why the
last decision was made**. For example: *"East + West → green by
priority (4 queued, waited 16 s, planned 10 s) · Last: North + South
ended by gap-out after 12 s"*.

When the run ends, a report shows each approach's green time, average
wait, maximum queue and a **phase timeline**.

---

## ⚙️ How It Works

```text
  4 camera feeds ──► YOLOv4-tiny ──► road zone filter ──► queue per approach
                                     (between 2 lines)          │
                                                                 ▼
   video of each approach ◄── lights ◄── adaptive controller ◄──┘
   (green plays, red paused)    North+South ⇄ East+West
                                GREEN → YELLOW → ALL-RED → next phase
```

### Real-world signal rules

| Rule | How it's applied |
|---|---|
| **Phases**: non-conflicting movements share the green | Two-phase plan: North + South, then East + West. Conflicting phases are never green at the same time. `--phasing four-phase` serves one approach at a time (split phasing) |
| **Yellow change interval** (ITE) | `yellow = 1 s + v / (2 × 3.05 m/s²)`, rounded up to 0.5 s and kept within 3–6 s. At 50 km/h this gives **3.5 s**; set it with `--speed-limit` |
| **All-red clearance** (ITE) | `all-red = (junction width + 5 m car) / v`. For a 20 m junction at 50 km/h this gives **2.0 s**; set it with `--intersection-width` |
| **Minimum green** | 10 s. With `--pedestrians` it becomes `7 s WALK + width / 1.2 m/s`, which is **24 s** for a 20 m crossing |
| **Maximum green** | 60 s, rising by 15 s after each max-out with cars left, up to 120 s |
| **Critical lane** | A phase's queue is its busiest approach, so the green is long enough for the longest queue in the phase |
| **No starvation** | No queued phase waits more than about 120 s |
| **Emergency preemption** | An emergency vehicle holds its phase's green, or ends the other phase's green and is served next |

### Signal logic (`controller.py`)

**Which phase gets green next**

```text
priority = queue + 0.1 × seconds waited on red
```

- Long queues go first, and a short queue that has waited long enough wins eventually.
- The phase that just had green isn't picked again while others are waiting.
- **Starvation guard:** once a queued phase has waited 120 s, the running green ends (after the minimum green) and the longest-waiting phase goes next.
- An **emergency vehicle** beats everything.
- If nobody is queued, phases are visited in rotation, so every camera keeps being checked.

**How long the green lasts**

```text
planned green = clamp(queue / 0.5 veh/s, 10 s, 60 s)
```

| The green ends when… | Name |
|---|---|
| every approach in the phase has been empty for 2 s and someone else is waiting | **gap-out** |
| the planned time is reached and someone else is waiting | **max-out** |
| another queued phase has waited 120 s | **max-wait** |
| an emergency vehicle appears on another phase (after 5 s) | **preempted** |
| the videos of every approach in the phase have finished | **cleared** |

If nobody else is waiting, the green simply stays on ("rest in green").

| Parameter | Value |
|---|---|
| Saturation (discharge) rate | 0.5 veh/s per approach |
| Min / max green | 10 s / 60 s (cap escalates +15 s, up to 120 s) |
| Yellow / all-red | 3.5 s / 2.0 s (50 km/h, 20 m junction) |
| Gap-out time | 2 s |
| Starvation limit | 120 s |
| Fixed-timer baseline | 30 s green per phase, same phases and change intervals |

The signal clock runs on **video time**, where one tick is one frame. The
timings are therefore exact no matter how fast detection runs on your
machine. The queue is the median of the last 5 detections, so a single
missed or doubled detection can't flip a decision.

### Detection area (`road_roi.py`)

You draw two lines per camera along the edges of the **lanes heading into
the intersection**. Each line is extended across the whole frame, and a
vehicle counts only if its **ground point** (the bottom-centre of its box,
where the wheels touch the road) lies between the lines. Traffic leaving
on the other carriageway, pavements, side roads and cars parked off the
road are all ignored. The lines are saved in `road_roi.json` in
resolution-independent coordinates.

On the sample highway clip, drawing the lines around the inbound lanes
dropped each camera's count from 35 to 21 vehicles. About 40% of what the
whole frame saw was traffic going the other way.

**Parked cars inside the lines** are handled by watching movement:

- A vehicle that hasn't moved more than 12 px during 20 s of *playing* footage is marked parked. Playing footage means its approach had green and it still didn't move.
- Parked vehicles are left out of the queue and the count.
- Paused (red) footage doesn't advance this clock, so cars waiting at a red are never mistaken for parked ones.
- A vehicle is only **counted** once it actually travels, so a car that just sits in view never adds to the count.

---

## 📊 Validation

Both validations use the **same `controller.py`** as the camera demo.
Both policies run the same two-phase plan and change intervals, so the
only thing that differs is the controller.

### Queue simulation (`simulate_comparison.py`)

This is a second-by-second simulation: 30 minutes of random (Poisson)
arrivals, with demand shares taken from the last camera run, and identical
arrivals for both policies.

| Metric | Fixed timer | Adaptive |
|---|---:|---:|
| Average delay per vehicle | 17.6 s | 14.8 s |
| Average queue (all approaches) | 6.6 | 5.5 |
| Longest queue | 19.0 | 13.5 |
| Time to clear all traffic | 1806 s | 1816 s |

### SUMO validation (`sumo_sim/traci_control.py`)

SUMO is driven one second at a time. Every step it counts the vehicles a
camera would see on each approach (everything within 100 m of the stop
line, moving or stopped), passes those counts to the controller, and sets
the light state directly.

| Metric | Fixed timer | Adaptive |
|---|---:|---:|
| Completed trips | 702 | 702 |
| Average waiting time | 11.8 s | 7.9 s (−33%) |
| Average trip duration | 114.3 s | 110.1 s |
| Average time loss | 24.5 s | 20.3 s (−17%) |

The detection range matters, and this is worth knowing:

| SUMO measurement | Fixed wait | Adaptive wait |
|---|---:|---:|
| Halted cars only | 11.8 s | 13.3 s (worse) |
| All cars within 60 m | 11.8 s | 8.7 s |
| **All cars within 100 m (default)** | 11.8 s | **7.9 s** |
| All cars within 150 m | 11.8 s | 6.7 s |

With halted cars only, the controller ends a green the moment the standing
queue clears, even while more cars are driving up. Those cars then hit a
red. Counting every car in view, as the camera does, avoids this.

> With two-phase operation the junction has twice the capacity, so at
> this demand level traffic is light and the gains are modest. These
> results come from the project's simulation scenarios (demand shares
> from counts of `[38, 61, 12, 31]`) and are not a guarantee of
> real-world improvement.

### On the camera footage

These runs used the four sample highway clips, with lines around the
inbound lanes:

| Plan | Time to clear all approaches |
|---|---:|
| Adaptive, two-phase | **57 s** |
| Fixed timer, two-phase | 60 s |
| Adaptive, four-phase (one road at a time) | 119 s |

---

## 🚨 Emergency Vehicle Preemption

YOLOv4-tiny's COCO classes have no "ambulance" or "fire truck", so
`emergency.py` looks for a **flashing red/blue light bar** on tracked
vehicles inside the road zone instead. When one is seen, an emergency on
a green approach **holds its phase's green**, and an emergency on a red
approach **ends the current green**, after which its phase is served next.

This is a heuristic, not a trained classifier, and the sample videos
contain no emergency vehicles. Because a red approach's video is paused,
a flashing light can only be seen on footage that is playing. With live
cameras, every approach would be watched continuously.

```bash
python -m pytest test_emergency.py
```

---

## 🖥️ Dashboard

```bash
streamlit run dashboard.py
```

The dashboard has four tabs:

- **Camera run:** the phase plan and change intervals, per-approach results, a phase timeline, and a decision log showing why each phase got green and why the green ended
- **Queues & lights:** each approach's queue over time and a strip showing the signal state
- **Queue simulation** and **SUMO validation:** fixed timer vs adaptive

---

## 🚀 Quick Start

### Run it in Google Colab (no install)

Open **[colab_demo.ipynb in Colab](https://colab.research.google.com/github/diyorbek57/Adaptive-Traffic-Signal-Control-Using-Computer-Vision/blob/main/colab_demo.ipynb)** and choose
*Runtime → Run all*. The notebook does the following:
- clones the repo and runs the tests
- previews the road lines on each camera
- runs the adaptive and fixed-timer controllers on the videos, then shows the recorded grid and summaries
- runs the queue simulation and the SUMO check

It can also take your own videos and open the dashboard. The whole notebook
takes about 5 minutes on the free CPU runtime.

### 1. Install

```bash
git clone https://github.com/SandeshBansode/Adaptive-Traffic-Signal-Control-Using-Computer-Vision.git
cd Adaptive-Traffic-Signal-Control-Using-Computer-Vision
pip install -r requirements.txt
```

OpenCV is pinned to 4.x, because OpenCV 5 removed the Darknet importer
that YOLOv4-tiny needs.

### 2. Add your videos

Put one video per approach in `videos/` as `1.mp4` … `4.mp4`, in the
order **North, East, South, West**. Straight road views work best, with
the camera looking along the road.

### 3. Draw the road lines

```bash
python calibrate_roi.py            # all four videos
python calibrate_roi.py videos/2.mp4   # or just one
```

1. Click two points on one edge of the **lanes heading into the junction**, then two points on their other edge. Leave out the opposite carriageway.
2. The vehicles detected on that frame are previewed: green ones are counted and grey ones are ignored.
3. Keys: `S` save, `U` undo, `R` reset, `C` copy the previous camera's lines, `N` skip, `Q` quit.

If a camera has no lines, `detect_and_signal.py` opens this tool
automatically before the run. Use `--skip-calibration` to run without
lines.

### 4. Run the intersection

```bash
python detect_and_signal.py                        # adaptive, two-phase
python detect_and_signal.py --mode fixed           # fixed-timer baseline, same videos
python detect_and_signal.py --phasing four-phase   # one approach at a time
python detect_and_signal.py --pedestrians          # pedestrian-safe minimum green
python detect_and_signal.py --speed-limit 70 --intersection-width 30
python detect_and_signal.py --loop --fast          # loop footage as continuous traffic, unthrottled
```

| Key | Action |
|---|---|
| `P` | Pause |
| `F` | Toggle fast mode |
| `Q` | Quit |

Other options: `--duration SECONDS`, `--min-green`, `--max-green`,
`--fixed-green`, `--no-record` and `--videos A B C D`. `--headless` runs without any windows, for servers or Colab:
it runs at full speed and saves the video and the summary image.

The run writes:

- `outputs/latest_run.json`: per-approach stats and the full phase log
- `outputs/telemetry.csv`: queue and light per approach, every 0.5 s
- `outputs/traffic_detection_output.mp4`: a recording of the window
- `outputs/summary.png`: the end-of-run report
- `out.txt`: vehicles counted per approach, used as demand by the simulations

### 5. Simulations and dashboard

```bash
python simulate_comparison.py
python sumo_sim/generate_routes.py
python sumo_sim/traci_control.py --mode both     # add --gui to watch (adaptive only: --mode adaptive --gui)
streamlit run dashboard.py
```

### Tests

```bash
python -m pytest
```

---

## 🧩 Project Structure

```text
Adaptive-Traffic-Signal-Control-Using-Computer-Vision/
│
├── detect_and_signal.py       # Camera demo: detection → queues → controller → lights → video
├── colab_demo.ipynb           # Google Colab notebook: runs the whole project in the browser
├── controller.py              # Adaptive + fixed-time signal controllers (shared by everything)
├── road_roi.py                # Detection area between two road-edge lines
├── calibrate_roi.py           # Tool to draw each camera's road lines
├── ui.py                      # OpenCV drawing: tiles, signal heads, report
├── emergency.py               # Flashing-light emergency heuristic
├── simulate_comparison.py     # Queue simulation, fixed vs adaptive
├── dashboard.py               # Streamlit dashboard
├── road_roi.json              # Saved road lines (created by calibrate_roi.py)
│
├── test_controller.py         # Signal logic and real-world timing tests
├── test_parked.py             # Parked-car filter tests
├── test_road_roi.py           # Road zone tests
├── test_tracker.py            # Centroid tracker tests
├── test_emergency.py          # Emergency heuristic tests
│
├── tracking/                  # Centroid tracker
├── models/                    # YOLOv4-tiny cfg / weights / COCO names
├── videos/                    # One video per approach (1.mp4 … 4.mp4)
├── sumo_sim/                  # SUMO network, route generator, TraCI controller loop
├── outputs/                   # Generated results
└── README.md
```

---

## 🛠️ Tech Stack

**Computer Vision**

`Python` · `YOLOv4-tiny` · `OpenCV DNN`

**Traffic Intelligence**

`Actuated Signal Control` · `Saturation Flow` · `Gap-out / Max-out`
· `Aging / Anti-starvation` · `Road-zone Counting`

**Simulation**

`SUMO` · `TraCI`

**Visualization**

`Streamlit` · `Matplotlib` · `Pandas`

---

## 🎓 Project Context

**Institution**

P.E.S. College of Engineering  
Department of Computer Science and Engineering

**Project**

Adaptive Traffic Signal Control Using Computer Vision

**Team**

Sandesh Bansode · Roll No. 49  
Syed Asimuddin · Roll No. 52

### Planned modules

- [x] Vehicle Detection
- [x] Traffic Analysis
- [x] Emergency Vehicle Detection
- [x] Signal Optimization
- [x] Traffic Simulation

---

## 🔒 License

**© 2026 Sandesh Bansode — All Rights Reserved**

This repository is publicly available for viewing and evaluation.

No permission is granted to copy, reproduce, redistribute, modify,
commercially exploit, or create derivative works from the original project
code without prior written permission from the copyright holder.

Third-party libraries, models, datasets, frameworks, and other external
materials remain subject to their respective licenses.

---

<div align="center">

### Built with Python, Computer Vision & Traffic Intelligence

**Adaptive Traffic Signal Control Using Computer Vision**

</div>
