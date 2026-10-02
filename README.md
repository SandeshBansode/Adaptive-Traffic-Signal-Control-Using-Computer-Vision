<div align="center">

# 🚦 Adaptive Traffic Signal Control
### Computer Vision • Intelligent Signal Timing • Traffic Simulation

**A computer-vision system that detects real-time traffic demand across four lanes and dynamically allocates green-light time using an adaptive signal-control algorithm.**

<br>

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLOv4--tiny-00FFFF?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-DNN-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![SUMO](https://img.shields.io/badge/SUMO-TraCI-222222?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

<br><br>

**P.E.S. College of Engineering · Computer Science & Engineering**

**Sandesh Bansode**

</div>

---

## ✦ Overview

Traditional fixed-time traffic signals allocate the same green time regardless of
the actual traffic demand. This project replaces that approach with a
**computer-vision-driven adaptive controller**.

The system:

- Detects vehicles from four lane-specific video feeds using **YOLOv4-tiny**
- Tracks vehicles with a lightweight **centroid tracker**
- Calculates lane demand from detected vehicles and carried-over backlog
- Dynamically allocates green-light duration
- Carries unserved traffic into the next cycle
- Uses an aging mechanism to prevent lane starvation
- Includes heuristic emergency-vehicle preemption
- Validates the approach through multi-cycle simulation and **SUMO**

---

## ◉ Demo

### Live 4-Lane Detection

All four approaches are processed simultaneously, with bounding boxes and
live vehicle counts rendered directly on the video.

<p align="center">
  <img src="images/live-detection.jpg" alt="Live four-lane traffic detection" width="900">
</p>

### Adaptive Signal Allocation

After detection, the controller converts traffic demand into lane-specific
green-light durations, served vehicles, and carry-over demand.

<p align="center">
  <img src="images/signal-summary.jpg" alt="Adaptive traffic signal summary" width="900">
</p>

### Lane-wise Results

The resulting plan exposes the allocation for every lane.

<p align="center">
  <img src="images/signal-results.jpg" alt="Lane-wise adaptive signal results" width="900">
</p>

### Fixed Timer vs Adaptive

A multi-cycle simulation compares the fixed 30-second policy with the adaptive
controller.

<p align="center">
  <img src="images/comparison-chart.jpg" alt="Fixed timer versus adaptive signal control" width="900">
</p>

---

## ⚙️ How It Works

```text
             ┌──────────────────────┐
             │  4 Lane Video Feeds  │
             └──────────┬───────────┘
                        ↓
             ┌──────────────────────┐
             │    YOLOv4-tiny       │
             │  Vehicle Detection   │
             └──────────┬───────────┘
                        ↓
             ┌──────────────────────┐
             │   Centroid Tracker   │
             │ Vehicle Identification│
             └──────────┬───────────┘
                        ↓
             ┌──────────────────────┐
             │    Lane Demand       │
             │ Count + Carry-over   │
             └──────────┬───────────┘
                        ↓
             ┌──────────────────────┐
             │ Adaptive Signal Plan │
             │ Saturation + Aging   │
             └──────────┬───────────┘
                        ↓
             ┌──────────────────────┐
             │ Green Time per Lane  │
             └──────────┬───────────┘
                        ↓
             ┌──────────────────────┐
             │ Next Cycle / SUMO    │
             └──────────────────────┘
```

### Core signal model

The controller uses:

```text
Demand = live vehicle count + carried-over backlog

Green Time =
clamp(Demand / Saturation Rate,
      Minimum Green,
      Maximum Green)
```

Current baseline parameters:

```text
Saturation rate       = 0.5 vehicles/sec
Minimum green         = 10 sec
Maximum green         = 60 sec
Cap escalation step   = 15 sec
Absolute maximum      = 120 sec
```

If a lane cannot clear its entire queue within its allocated phase, the
remaining demand is **carried into the next cycle** instead of being discarded.

---

## ✦ Key Features

| Capability | Implementation |
|---|---|
| Vehicle detection | YOLOv4-tiny + OpenCV DNN |
| Vehicle tracking | Centroid tracker |
| Lane processing | 4 simultaneous video feeds |
| Signal timing | Saturation-flow adaptive model |
| Queue persistence | Carry-over backlog |
| Fairness | Aging / anti-starvation |
| Emergency handling | Red/blue light-bar heuristic |
| Comparison | Multi-cycle fixed vs adaptive simulation |
| Traffic simulation | SUMO + TraCI |
| Monitoring | Streamlit dashboard |

---

## 📊 Validation

### Multi-cycle simulation

The project includes a 25-cycle simulation using the same traffic-demand
pattern for both policies.

**Sample result from the project simulation:**

| Metric | Fixed Timer | Adaptive |
|---|---:|---:|
| Average queued vehicles / cycle | 587.4 | 8.7 |
| Final backlog | 1117.0 | 0.0 |

The tested simulation reported an approximately **98.5% reduction in average
queue length**.

> These values are results from the project's simulation scenario and should
> not be interpreted as a guaranteed real-world improvement.

### SUMO validation

The same signal-planning function is also evaluated inside a SUMO microscopic
traffic simulation.

| Metric | Fixed | Adaptive |
|---|---:|---:|
| Completed trips | 702 | 702 |
| Average waiting time | 131.6s | 53.4s |
| Average duration | 256.3s | 165.7s |
| Average time loss | 166.5s | 75.9s |

The tested SUMO scenario reported an approximately **59.4% reduction in
average waiting time**.

---

## 🚨 Emergency Vehicle Preemption

YOLOv4-tiny's COCO classes do not directly identify ambulances or fire trucks.

Instead, `emergency.py` uses a visual heuristic that looks for a **flashing
red/blue light bar** on tracked vehicles.

When detected, the corresponding lane can receive signal preemption.

This is intentionally documented as a **heuristic**, not a trained
emergency-vehicle classifier. The included sample videos do not contain real
emergency vehicles.

Test the detection logic with:

```bash
python test_emergency.py
```

---

## 🖥️ Dashboard

The project includes a Streamlit dashboard for inspecting:

- Current signal plan
- Green time
- Vehicles served
- Carried-over demand
- Cycles waited
- Vehicle-count history
- Persisted backlog
- Fixed-vs-adaptive comparison
- Simulation metrics

Launch it with:

```bash
streamlit run dashboard.py
```

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/SandeshBansode/Adaptive-Traffic-Signal-Control-Using-Computer-Vision.git
cd Adaptive-Traffic-Signal-Control-Using-Computer-Vision
```

### 2. Install dependencies

```bash
pip install opencv-python numpy scipy matplotlib streamlit pandas
```

For SUMO validation, install the SUMO/TraCI dependencies required by your
environment.

### 3. Run live detection

```bash
python detect_and_signal.py
```

The application opens:

```text
Adaptive Traffic Signal - Live Detection (4 Lanes)
```

Controls:

| Key | Action |
|---|---|
| `q` | Quit |
| `p` | Pause |
| Any key | Close final summary |

### 4. Run the comparison

```bash
python simulate_comparison.py
```

### 5. Launch the dashboard

```bash
streamlit run dashboard.py
```

### 6. Run SUMO validation

```bash
python sumo_sim/generate_routes.py
python sumo_sim/traci_control.py --mode both
```

For the visual adaptive simulation:

```bash
python sumo_sim/traci_control.py --mode adaptive --gui
```

---

## 🧩 Project Structure

```text
Adaptive-Traffic-Signal-Control-Using-Computer-Vision/
│
├── detect_and_signal.py       # Main computer-vision pipeline
├── emergency.py               # Emergency vehicle heuristic
├── test_emergency.py          # Emergency detection test
├── simulate_comparison.py     # Fixed vs adaptive simulation
├── dashboard.py               # Streamlit dashboard
│
├── images/
│   ├── live-detection.jpg
│   ├── signal-summary.jpg
│   ├── signal-results.jpg
│   └── comparison-chart.jpg
│
├── tracking/
│   ├── centroidtracker.py
│   └── trackableobject.py
│
├── models/
│   ├── yolov4-tiny.cfg
│   ├── yolov4-tiny.weights
│   └── coco.names
│
├── videos/                    # Four lane video inputs
│
├── sumo_sim/
│   ├── generate_routes.py
│   ├── traci_control.py
│   └── SUMO network/config files
│
├── outputs/                   # Generated results
├── out.txt
├── signal_state.json
└── README.md
```

---

## 🛠️ Tech Stack

**Computer Vision**

`Python` · `YOLOv4-tiny` · `OpenCV DNN`

**Traffic Intelligence**

`Adaptive Signal Control` · `Saturation Flow` · `Carry-over Queue`
· `Aging / Anti-starvation`

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
