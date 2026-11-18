"""
Analytics dashboard for the Adaptive Traffic Signal Control project.

Reads the files detect_and_signal.py, simulate_comparison.py and
sumo_sim/traci_control.py write to outputs/ - it doesn't run detection
itself, so it stays lightweight and can be left open in a browser tab.

Run with:
    streamlit run dashboard.py
"""

import os
import json
import time

import altair as alt
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

LATEST_RUN_PATH = os.path.join(OUTPUTS_DIR, "latest_run.json")
TELEMETRY_PATH = os.path.join(OUTPUTS_DIR, "telemetry.csv")
COMPARISON_JSON_PATH = os.path.join(OUTPUTS_DIR, "comparison.json")
COMPARISON_CHART_PATH = os.path.join(OUTPUTS_DIR, "comparison_chart.png")
SUMO_COMPARISON_PATH = os.path.join(OUTPUTS_DIR, "sumo_comparison.json")
ROAD_ROI_PATH = os.path.join(BASE_DIR, "road_roi.json")

ADAPTIVE_COLOR = "#2e9e4f"
FIXED_COLOR = "#9aa0a6"
EMERGENCY_COLOR = "#d63fd6"
LIGHT_COLORS = {"green": "#2e9e4f", "yellow": "#f5a524", "red": "#e5484d"}
POLICY_SCALE = alt.Scale(domain=["Fixed timer", "Adaptive"], range=[FIXED_COLOR, ADAPTIVE_COLOR])

st.set_page_config(page_title="Adaptive Traffic Signal", page_icon="🚦", layout="wide")


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (ValueError, OSError):
        return None


def time_ago(timestamp):
    seconds = max(0, time.time() - timestamp)
    if seconds < 60:
        return "{:.0f} s ago".format(seconds)
    if seconds < 3600:
        return "{:.0f} min ago".format(seconds / 60)
    if seconds < 86400:
        return "{:.1f} h ago".format(seconds / 3600)
    return "{:.0f} days ago".format(seconds / 86400)


def pct_reduction(before, after):
    return (before - after) / before * 100 if before else 0.0


def mmss(seconds):
    mins, secs = divmod(int(seconds), 60)
    return "{}:{:02d}".format(mins, secs)


def empty_state(message, command):
    st.info("{}\n\nRun `{}`, then refresh.".format(message, command))


def policy_legend():
    return alt.Legend(orient="bottom", title=None)


# --- Sidebar ----------------------------------------------------------------

with st.sidebar:
    st.header("🚦 Controls")
    auto_refresh = st.toggle("Auto-refresh", value=False,
                             help="Re-read the output files periodically.")
    refresh_every = st.slider("Interval (s)", 2, 30, 5, disabled=not auto_refresh)
    if st.button("Refresh now", width="stretch"):
        st.rerun()

    st.divider()
    st.subheader("Data files")
    for name, path in [
        ("Road lines", ROAD_ROI_PATH),
        ("Latest camera run", LATEST_RUN_PATH),
        ("Queue telemetry", TELEMETRY_PATH),
        ("Queue simulation", COMPARISON_JSON_PATH),
        ("SUMO validation", SUMO_COMPARISON_PATH),
    ]:
        if os.path.exists(path):
            st.markdown("✅ {}  \n<small>updated {}</small>".format(name, time_ago(os.path.getmtime(path))),
                        unsafe_allow_html=True)
        else:
            st.markdown("⚪ {}  \n<small>not generated yet</small>".format(name), unsafe_allow_html=True)

    st.divider()
    st.caption("Generate data with:")
    st.code("python calibrate_roi.py\n"
            "python detect_and_signal.py\n"
            "python simulate_comparison.py\n"
            "python sumo_sim/traci_control.py --mode both", language="bash")


# --- Tabs -------------------------------------------------------------------

def render_run_tab(run):
    if run is None or "lanes" not in run or (run["phases"] and "lanes" not in run["phases"][0]):
        empty_state("No camera run in the current format yet.", "python detect_and_signal.py")
        return

    lanes = run["lanes"]
    phases = run["phases"]
    labels = run["labels"]
    sim_time = run["sim_time"]

    if not all(lane.get("road_lines", True) for lane in lanes):
        missing = [lane["label"] for lane in lanes if not lane.get("road_lines", True)]
        st.warning("No road lines for {} - those cameras counted the whole frame. "
                   "Run `python calibrate_roi.py`.".format(", ".join(missing)))

    settings = run.get("settings", {})
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Control mode", run["mode"].capitalize(), settings.get("phasing"), delta_color="off", border=True)
    k2.metric("Simulated time", mmss(sim_time), border=True)
    k3.metric("Greens", len(phases), border=True)
    k4.metric("Vehicles counted", sum(lane["counted"] for lane in lanes), border=True)

    plan = run.get("phase_plan", [])
    if plan:
        st.caption("Phase plan: {} · yellow {:.1f} s · all-red {:.1f} s · min green {} · {} km/h, {} m wide{}".format(
            " → ".join(p["name"] for p in plan), settings.get("yellow", 0), settings.get("all_red", 0),
            "{:.0f} s".format(settings["min_green"]) if settings.get("min_green") else "-",
            settings.get("speed_limit_kmh"), settings.get("intersection_width_m"),
            " · pedestrian-safe" if settings.get("pedestrians") else ""))

    st.subheader("Per-approach results")
    cols = st.columns(len(lanes))
    for col, lane in zip(cols, lanes):
        with col.container(border=True):
            st.markdown("#### {}".format(lane["label"]))
            st.metric("Green time", "{:.0f} s".format(lane["green_time"]),
                      "{} phase{}".format(lane["phases"], "" if lane["phases"] == 1 else "s"), delta_color="off")
            st.progress(min(1.0, lane["green_time"] / sim_time) if sim_time else 0.0,
                        text="{:.0f}% of the time".format(lane["green_time"] / sim_time * 100 if sim_time else 0))
            st.caption("counted **{}** · max queue **{}** · avg queue {:.1f}".format(
                lane["counted"], lane["max_queue"], lane["avg_queue"]))
            st.caption("average wait before green: **{}**".format(
                "{:.0f} s".format(lane["avg_wait"]) if lane["avg_wait"] is not None else "-"))

    st.subheader("Phase timeline")
    if phases:
        # A phase greens every approach in it - one bar per approach
        timeline = pd.DataFrame([{
            "Approach": labels[lane],
            "Phase": p.get("label"),
            "start": p["start"],
            "end": p["start"] + (p["green"] if p["green"] is not None else max(0.0, sim_time - p["start"])),
            "Green (s)": p["green"],
            "Queue at start": p["queue_at_start"],
            "Chosen by": p.get("choice"),
            "Ended by": p.get("end_reason") or "end of run",
            "Kind": "Emergency" if p.get("emergency") else "Normal",
        } for p in phases for lane in p["lanes"]])
        chart = alt.Chart(timeline).mark_bar(cornerRadius=2, height=22).encode(
            x=alt.X("start:Q", title="Seconds"),
            x2="end:Q",
            y=alt.Y("Approach:N", title=None, sort=labels),
            color=alt.Color("Phase:N", legend=alt.Legend(orient="bottom", title=None)),
            stroke=alt.condition(alt.datum.Kind == "Emergency", alt.value(EMERGENCY_COLOR), alt.value(None)),
            tooltip=["Phase", "Approach", "start", "Green (s)", "Queue at start", "Chosen by", "Ended by"],
        ).properties(height=40 * len(labels) + 30)
        st.altair_chart(chart, use_container_width=True)

        with st.expander("Decision log - why each phase got green and why it ended"):
            log = pd.DataFrame([{
                "Start (s)": p["start"],
                "Phase": p.get("label"),
                "Chosen by": p.get("choice"),
                "Queues": ", ".join("{} {}".format(labels[lane], q)
                                    for lane, q in zip(p["lanes"], p.get("lane_queues", []))),
                "Waited (s)": p["waited"],
                "Planned (s)": p["planned"],
                "Green (s)": p["green"],
                "Ended by": p.get("end_reason") or "end of run",
            } for p in phases])
            st.dataframe(log, hide_index=True, width="stretch")
    else:
        st.info("No green phases recorded.")


def render_queue_tab():
    if not os.path.exists(TELEMETRY_PATH):
        empty_state("No queue telemetry yet.", "python detect_and_signal.py")
        return
    try:
        df = pd.read_csv(TELEMETRY_PATH)
    except (ValueError, OSError):
        st.warning("telemetry.csv could not be read (it may be mid-write) - refresh in a moment.")
        return
    if df.empty or not {"t", "lane", "light", "queue"}.issubset(df.columns):
        empty_state("telemetry.csv is from an older version.", "python detect_and_signal.py")
        return

    st.subheader("Queue in each approach's road zone")
    st.caption("A red approach's video is paused, so its queue holds still until it gets green.")
    queue_chart = alt.Chart(df).mark_line(interpolate="step-after", strokeWidth=2).encode(
        x=alt.X("t:Q", title="Seconds"),
        y=alt.Y("queue:Q", title="Vehicles queued"),
        color=alt.Color("lane:N", title=None, legend=alt.Legend(orient="bottom")),
        tooltip=["lane", "t", "queue", "light"],
    ).properties(height=320)
    st.altair_chart(queue_chart, use_container_width=True)

    st.subheader("Signal state")
    light_chart = alt.Chart(df).mark_rect().encode(
        x=alt.X("t:Q", title="Seconds", bin=alt.Bin(step=0.5)),
        y=alt.Y("lane:N", title=None),
        color=alt.Color("light:N", scale=alt.Scale(domain=list(LIGHT_COLORS), range=list(LIGHT_COLORS.values())),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["lane", "t", "light", "queue"],
    ).properties(height=40 * df["lane"].nunique() + 30)
    st.altair_chart(light_chart, use_container_width=True)


def render_simulation_tab():
    comparison = load_json(COMPARISON_JSON_PATH)
    if comparison is None or "delay_improvement_pct" not in comparison:
        empty_state("No queue simulation yet.", "python simulate_comparison.py")
        return

    fixed, adaptive = comparison["fixed_timer"], comparison["adaptive"]
    st.caption("Second-by-second queue simulation with the same controllers as the camera demo: "
               "{} vehicles over {} s, arrival rates {} veh/h per approach.".format(
                   comparison["total_vehicles"], comparison["demand_duration"], comparison["arrival_rates_vph"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average delay", "{:.0f} s".format(adaptive["avg_delay"]),
              "-{:.0f}% vs fixed".format(comparison["delay_improvement_pct"]), delta_color="inverse", border=True)
    c2.metric("Average queue", "{:.1f}".format(adaptive["avg_queue"]),
              "-{:.0f}% vs fixed".format(comparison["queue_improvement_pct"]), delta_color="inverse", border=True)
    c3.metric("Longest queue", "{:.0f}".format(adaptive["max_queue"]),
              "{:+.0f} vs fixed".format(adaptive["max_queue"] - fixed["max_queue"]),
              delta_color="inverse", border=True)
    c4.metric("Cleared after", mmss(adaptive["clearance_time"]),
              "{:+.0f} s vs fixed".format(adaptive["clearance_time"] - fixed["clearance_time"]),
              delta_color="inverse", border=True)

    st.subheader("Total queue over time")
    rows = []
    for policy, result in (("Fixed timer", fixed), ("Adaptive", adaptive)):
        series = result["queue_over_time"]
        step = max(1, len(series) // 600)
        rows += [{"Seconds": t, "Queued vehicles": q, "Policy": policy}
                 for t, q in enumerate(series) if t % step == 0]
    chart = alt.Chart(pd.DataFrame(rows)).mark_line(strokeWidth=2).encode(
        x="Seconds:Q", y="Queued vehicles:Q",
        color=alt.Color("Policy:N", scale=POLICY_SCALE, legend=policy_legend()),
        tooltip=["Policy", "Seconds", "Queued vehicles"],
    ).properties(height=360)
    st.altair_chart(chart, use_container_width=True)


def render_sumo_tab():
    sumo = load_json(SUMO_COMPARISON_PATH)
    if sumo is None or "fixed" not in sumo or "adaptive" not in sumo:
        empty_state("No SUMO comparison yet.", "python sumo_sim/traci_control.py --mode both")
        return

    fixed, adaptive = sumo["fixed"], sumo["adaptive"]
    st.caption("SUMO microsimulation driven second by second through TraCI by the same controllers "
               "as the camera demo.")

    metrics = [
        ("Avg waiting time", "avg_waiting_time"),
        ("Avg trip duration", "avg_duration"),
        ("Avg time loss", "avg_time_loss"),
    ]
    cols = st.columns(len(metrics) + 1)
    for col, (title, key) in zip(cols, metrics):
        col.metric(title, "{:.1f} s".format(adaptive[key]),
                   "-{:.1f}% vs fixed".format(pct_reduction(fixed[key], adaptive[key])),
                   delta_color="inverse", border=True)
    cols[-1].metric("Completed trips", adaptive["completed_trips"],
                    "{:+d} vs fixed".format(adaptive["completed_trips"] - fixed["completed_trips"]), border=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Per-vehicle averages")
        bars_df = pd.DataFrame(
            [{"Metric": title, "Policy": "Fixed timer", "Seconds": fixed[key]} for title, key in metrics]
            + [{"Metric": title, "Policy": "Adaptive", "Seconds": adaptive[key]} for title, key in metrics]
        )
        chart = alt.Chart(bars_df).mark_bar().encode(
            x=alt.X("Seconds:Q"),
            y=alt.Y("Policy:N", title=None, axis=None),
            row=alt.Row("Metric:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
            color=alt.Color("Policy:N", scale=POLICY_SCALE, legend=policy_legend()),
            tooltip=["Metric", "Policy", alt.Tooltip("Seconds:Q", format=".1f")],
        ).properties(height=50)
        st.altair_chart(chart, use_container_width=True)

    with right:
        st.subheader("Halted vehicles over time")
        rows = []
        for policy, stats in (("Fixed timer", fixed), ("Adaptive", adaptive)):
            for entry in stats.get("queue_log", []):
                rows.append({"Seconds": entry["t"], "Policy": policy, "Halted vehicles": sum(entry["queues"])})
        if rows:
            chart = alt.Chart(pd.DataFrame(rows)).mark_line(strokeWidth=2).encode(
                x="Seconds:Q", y="Halted vehicles:Q",
                color=alt.Color("Policy:N", scale=POLICY_SCALE, legend=policy_legend()),
                tooltip=["Policy", "Seconds", "Halted vehicles"],
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("Re-run the SUMO validation to record queues over time.")


def render_dashboard():
    run = load_json(LATEST_RUN_PATH)

    st.title("🚦 Adaptive Traffic Signal Control")
    if run is not None and "timestamp" in run:
        st.caption("Last camera run {} · reads the files in outputs/ written by the project scripts".format(
            time_ago(run["timestamp"])))
    else:
        st.caption("Reads the files in outputs/ written by the project scripts")

    run_tab, queue_tab, sim_tab, sumo_tab = st.tabs(
        ["Camera run", "Queues & lights", "Queue simulation", "SUMO validation"]
    )
    with run_tab:
        render_run_tab(run)
    with queue_tab:
        render_queue_tab()
    with sim_tab:
        render_simulation_tab()
    with sumo_tab:
        render_sumo_tab()


st.fragment(render_dashboard, run_every=refresh_every if auto_refresh else None)()
