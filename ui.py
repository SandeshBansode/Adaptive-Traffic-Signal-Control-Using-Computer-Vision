"""
Drawing helpers for the OpenCV windows in detect_and_signal.py: the live
4-approach grid (signal heads, road zone, vehicle boxes, status bar) and
the end-of-run summary with its phase timeline. Pure rendering - no
detection or signal logic.

All colors are BGR.
"""

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_BOLD = cv2.FONT_HERSHEY_TRIPLEX

BG = (28, 24, 22)
PANEL = (44, 38, 34)
PANEL_LIGHT = (62, 54, 48)
TEXT = (240, 238, 236)
MUTED = (160, 150, 145)
GREEN = (90, 200, 70)
AMBER = (20, 175, 250)
RED = (70, 70, 235)
LAMP_OFF = (55, 55, 55)
EMERGENCY = (255, 60, 220)
EMERGENCY_ALT = (255, 140, 40)
BOX = (90, 220, 90)
BOX_OFF_ROAD = (150, 150, 150)
BOX_PARKED = (200, 170, 120)
ROAD_LINE = (60, 220, 255)

LIGHT_COLORS = {"green": GREEN, "yellow": AMBER, "red": RED}

# One color per approach, used for the phase timeline
LANE_COLORS = [(90, 200, 70), (230, 160, 60), (60, 160, 240), (200, 110, 220),
               (80, 200, 200), (120, 120, 240)]

HEADER_HEIGHT = 44
STATUS_BAR_HEIGHT = 64


def text(img, s, org, scale=0.5, color=TEXT, thickness=1, font=FONT):
    cv2.putText(img, s, org, font, scale, color, thickness, cv2.LINE_AA)


def text_size(s, scale=0.5, thickness=1, font=FONT):
    (w, h), _ = cv2.getTextSize(s, font, scale, thickness)
    return w, h


def blend_rect(img, pt1, pt2, color, alpha):
    """Fills a rectangle with `color` at the given opacity."""
    x1, y1 = max(0, pt1[0]), max(0, pt1[1])
    x2, y2 = min(img.shape[1], pt2[0]), min(img.shape[0], pt2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.empty_like(roi)
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def blend_poly(img, poly, color, alpha):
    overlay = img.copy()
    cv2.fillPoly(overlay, [poly], color, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)


def pill(img, s, x, y, bg, fg=(20, 20, 20), scale=0.45):
    """Draws a filled label badge with its top-left at (x, y); returns its width."""
    w, h = text_size(s, scale)
    pad_x, pad_y = 8, 6
    cv2.rectangle(img, (x, y), (x + w + 2 * pad_x, y + h + 2 * pad_y), bg, -1, cv2.LINE_AA)
    text(img, s, (x + pad_x, y + h + pad_y), scale, fg)
    return w + 2 * pad_x


def progress_bar(img, x, y, w, h, fraction, color, track=PANEL_LIGHT):
    cv2.rectangle(img, (x, y), (x + w, y + h), track, -1)
    fill = int(w * max(0.0, min(1.0, fraction)))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)


def vehicle_box(img, rect, label=None, color=BOX):
    """Thin box with bracketed corners and an optional ID tag."""
    sx, sy, ex, ey = rect
    cv2.rectangle(img, (sx, sy), (ex, ey), color, 1, cv2.LINE_AA)
    corner = max(4, min(14, (ex - sx) // 4, (ey - sy) // 4))
    for (cx, cy, dx, dy) in ((sx, sy, 1, 1), (ex, sy, -1, 1), (sx, ey, 1, -1), (ex, ey, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * corner, cy), color, 2, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * corner), color, 2, cv2.LINE_AA)
    if label:
        w, h = text_size(label, 0.38)
        top = max(0, sy - h - 6)
        cv2.rectangle(img, (sx, top), (sx + w + 6, top + h + 6), color, -1)
        text(img, label, (sx + 3, top + h + 3), 0.38, (20, 20, 20))


def off_road_box(img, rect):
    sx, sy, ex, ey = rect
    cv2.rectangle(img, (sx, sy), (ex, ey), BOX_OFF_ROAD, 1, cv2.LINE_AA)


def parked_box(img, rect):
    """Dashed box with a 'P' tag: inside the lines but not moving with traffic."""
    sx, sy, ex, ey = rect
    for (x1, y1, x2, y2) in ((sx, sy, ex, sy), (sx, ey, ex, ey), (sx, sy, sx, ey), (ex, sy, ex, ey)):
        length = max(abs(x2 - x1), abs(y2 - y1))
        for start in range(0, length, 8):
            end = min(length, start + 4)
            fx, fy = (x2 - x1) / max(1, length), (y2 - y1) / max(1, length)
            cv2.line(img, (int(x1 + fx * start), int(y1 + fy * start)),
                     (int(x1 + fx * end), int(y1 + fy * end)), BOX_PARKED, 1, cv2.LINE_AA)
    w, h = text_size("P parked", 0.38)
    top = max(0, sy - h - 6)
    cv2.rectangle(img, (sx, top), (sx + w + 6, top + h + 6), BOX_PARKED, -1)
    text(img, "P parked", (sx + 3, top + h + 3), 0.38, (20, 20, 20))


def road_zone(img, roi, light):
    """Shades the counting zone and draws the two road-edge lines."""
    h, w = img.shape[:2]
    tint = {"green": GREEN, "yellow": AMBER}.get(light, RED)
    poly = roi.polygon(w, h)
    if len(poly) >= 3:
        blend_poly(img, poly, tint, 0.12)
    for p1, p2 in roi.edge_lines(w, h):
        cv2.line(img, p1, p2, ROAD_LINE, 2, cv2.LINE_AA)


def signal_head(img, x, y, light, lamp_radius=9):
    """Vertical 3-lamp traffic light; `light` is 'red', 'yellow' or 'green'."""
    pad = 5
    w = 2 * lamp_radius + 2 * pad
    h = 3 * (2 * lamp_radius) + 4 * pad
    cv2.rectangle(img, (x, y), (x + w, y + h), (15, 15, 15), -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 1, cv2.LINE_AA)
    for i, name in enumerate(("red", "yellow", "green")):
        center = (x + w // 2, y + pad + lamp_radius + i * (2 * lamp_radius + pad))
        on = name == light
        cv2.circle(img, center, lamp_radius, LIGHT_COLORS[name] if on else LAMP_OFF, -1, cv2.LINE_AA)
        if on:
            cv2.circle(img, center, lamp_radius + 2, LIGHT_COLORS[name], 1, cv2.LINE_AA)
    return w, h


def lane_tile(frame, label, light, light_text, queue, counted, boxes, roi,
              progress, green_progress=None, finished=False, emergency=False,
              flash_phase=0, notice=None):
    """Composes one approach's tile.

    boxes: list of (rect, on_road, object_id_or_None, is_emergency, parked)
    roi: the approach's RoadROI, or None if no road lines are configured
    green_progress: elapsed/planned fraction of the current green, or None
    """
    tile = frame.copy()
    h, w = tile.shape[:2]

    if light == "red":
        # Stopped traffic: dim the frozen frame so the moving approach stands out
        blend_rect(tile, (0, 0), (w, h), BG, 0.35)

    if roi is not None:
        road_zone(tile, roi, light)

    for rect, on_road, object_id, is_emergency, parked in boxes:
        if not on_road:
            off_road_box(tile, rect)
        elif parked:
            parked_box(tile, rect)
        elif is_emergency:
            vehicle_box(tile, rect, "#{} EMERGENCY".format(object_id), EMERGENCY)
        else:
            vehicle_box(tile, rect, "#{}".format(object_id) if object_id is not None else None)

    # Header: name, light state, queue / counted
    blend_rect(tile, (0, 0), (w, HEADER_HEIGHT), BG, 0.8)
    text(tile, label.upper(), (12, 29), 0.7, TEXT, 1, FONT_BOLD)
    x = 12 + text_size(label.upper(), 0.7, 1, FONT_BOLD)[0] + 12
    pill_bg = MUTED if finished else LIGHT_COLORS[light]
    x += pill(tile, light_text, x, 12, pill_bg) + 6
    if emergency:
        pill(tile, "EMERGENCY", x, 12, EMERGENCY if flash_phase % 2 == 0 else EMERGENCY_ALT)

    queue_str = str(queue)
    qw, _ = text_size(queue_str, 0.9, 2, FONT_BOLD)
    text(tile, queue_str, (w - qw - 12, 32), 0.9, TEXT, 2, FONT_BOLD)
    caption = "counted {}   queued".format(counted)
    cw, _ = text_size(caption, 0.42)
    text(tile, caption, (w - qw - cw - 20, 28), 0.42, MUTED)

    signal_head(tile, 10, HEADER_HEIGHT + 10, "red" if finished else light)

    # Footer: green countdown + video progress
    blend_rect(tile, (0, h - 30), (w, h), BG, 0.75)
    if notice:
        text(tile, notice, (12, h - 11), 0.42, AMBER)
    elif green_progress is not None:
        text(tile, "green", (12, h - 11), 0.42, MUTED)
        progress_bar(tile, 64, h - 19, 200, 8, green_progress, LIGHT_COLORS.get(light, GREEN))
    text(tile, "video", (w - 232, h - 11), 0.42, MUTED)
    progress_bar(tile, w - 180, h - 18, 168, 6, progress, MUTED if light == "red" else GREEN)

    if emergency:
        border = EMERGENCY if flash_phase % 2 == 0 else EMERGENCY_ALT
        cv2.rectangle(tile, (0, 0), (w - 1, h - 1), border, 4)
    elif light in ("green", "yellow") and not finished:
        cv2.rectangle(tile, (0, 0), (w - 1, h - 1), LIGHT_COLORS[light], 3)
    else:
        cv2.rectangle(tile, (0, 0), (w - 1, h - 1), (70, 70, 70), 1)
    return tile


def status_bar(width, sim_time, mode, phase_text, decision_text, fps, fast=False, paused=False):
    bar = np.full((STATUS_BAR_HEIGHT, width, 3), BG, dtype=np.uint8)
    cv2.line(bar, (0, 0), (width, 0), PANEL_LIGHT, 1)

    mins, secs = divmod(int(sim_time), 60)
    x = 14
    for key, value in (("Time", "{:02d}:{:02d}".format(mins, secs)), ("Mode", mode.upper())):
        text(bar, key, (x, 26), 0.45, MUTED)
        x += text_size(key, 0.45)[0] + 8
        text(bar, value, (x, 26), 0.55, TEXT)
        x += text_size(value, 0.55)[0] + 26
    text(bar, phase_text, (x, 26), 0.55, TEXT)
    x += text_size(phase_text, 0.55)[0] + 16
    if paused:
        pill(bar, "PAUSED", x, 9, AMBER)
    elif fast:
        pill(bar, "FAST", x, 9, AMBER)

    text(bar, decision_text, (14, 52), 0.45, MUTED)

    right = "{:.0f} fps   [P] pause  [F] fast  [Q] quit".format(fps)
    text(bar, right, (width - text_size(right, 0.45)[0] - 14, 26), 0.45, MUTED)
    return bar


def paused_overlay(grid):
    h, w = grid.shape[:2]
    blend_rect(grid, (0, 0), (w, h), BG, 0.55)
    card_w, card_h = 420, 130
    x, y = (w - card_w) // 2, (h - card_h) // 2
    cv2.rectangle(grid, (x, y), (x + card_w, y + card_h), PANEL, -1)
    cv2.rectangle(grid, (x, y), (x + card_w, y + card_h), AMBER, 2)
    s = "PAUSED"
    tw, _ = text_size(s, 1.3, 2, FONT_BOLD)
    text(grid, s, ((w - tw) // 2, y + 62), 1.3, TEXT, 2, FONT_BOLD)
    hint = "any key to resume   |   Q to quit"
    hw, _ = text_size(hint, 0.5)
    text(grid, hint, ((w - hw) // 2, y + 100), 0.5, MUTED)
    return grid


# --- Summary panel ----------------------------------------------------------

SUMMARY_WIDTH = 1100


def summary_panel(labels, lane_stats, phases, sim_time, mode, phasing="two-phase"):
    """End-of-run report: per-approach stats and the phase timeline."""
    n = len(labels)
    card_h = 92
    timeline_row = 30
    height = 120 + n * (card_h + 12) + 60 + n * timeline_row + 90
    panel = np.full((height, SUMMARY_WIDTH, 3), BG, dtype=np.uint8)

    mins, secs = divmod(int(sim_time), 60)
    text(panel, "Signal Control Report", (32, 50), 0.95, TEXT, 1, FONT_BOLD)
    subtitle = "{} control, {}   |   {:d}:{:02d} simulated   |   {} greens   |   {} vehicles counted".format(
        mode.capitalize(), phasing, mins, secs, len(phases), sum(s["counted"] for s in lane_stats))
    text(panel, subtitle, (32, 82), 0.5, MUTED)

    y = 108
    col_x = [300, 470, 640, 810]
    for i, (label, stats) in enumerate(zip(labels, lane_stats)):
        cv2.rectangle(panel, (24, y), (SUMMARY_WIDTH - 24, y + card_h), PANEL, -1)
        lane_color = LANE_COLORS[next((p["phase"] for p in phases if i in p["lanes"]), i) % len(LANE_COLORS)]
        cv2.rectangle(panel, (24, y), (30, y + card_h), lane_color, -1)
        text(panel, label, (48, y + 40), 0.75, TEXT, 1, FONT_BOLD)
        text(panel, "{} green{}".format(stats["phases"], "" if stats["phases"] == 1 else "s"),
             (48, y + 68), 0.45, MUTED)

        cells = [
            ("COUNTED", "{}".format(stats["counted"])),
            ("GREEN TIME", "{:.0f} s".format(stats["green_time"])),
            ("AVG WAIT", "{:.0f} s".format(stats["avg_wait"]) if stats["avg_wait"] is not None else "-"),
            ("MAX QUEUE", "{}".format(stats["max_queue"])),
        ]
        for x, (key, value) in zip(col_x, cells):
            text(panel, key, (x, y + 34), 0.38, MUTED)
            text(panel, value, (x, y + 66), 0.75, TEXT, 1)

        share = stats["green_time"] / sim_time if sim_time > 0 else 0
        text(panel, "GREEN SHARE", (col_x[-1] + 160, y + 34), 0.38, MUTED)
        progress_bar(panel, col_x[-1] + 160, y + 52, 100, 12, share, lane_color)
        text(panel, "{:.0f}%".format(share * 100), (col_x[-1] + 160, y + 82), 0.42, MUTED)
        y += card_h + 12

    # Phase timeline: one row per approach, green bars over time
    y += 16
    text(panel, "PHASE TIMELINE", (32, y), 0.42, MUTED)
    y += 14
    tl_x = 120
    tl_w = SUMMARY_WIDTH - tl_x - 32
    scale = tl_w / sim_time if sim_time > 0 else 0
    for i, label in enumerate(labels):
        row_y = y + i * timeline_row
        text(panel, label, (32, row_y + 19), 0.45, MUTED)
        cv2.rectangle(panel, (tl_x, row_y + 4), (tl_x + tl_w, row_y + timeline_row - 4), PANEL, -1)
    for phase in phases:
        green = phase["green"] if phase["green"] is not None else max(0.0, sim_time - phase["start"])
        x1 = tl_x + int(phase["start"] * scale)
        x2 = max(x1 + 2, tl_x + int((phase["start"] + green) * scale))
        # A phase greens every approach in it - draw it on each one's row
        for lane in phase["lanes"]:
            row_y = y + lane * timeline_row
            color = EMERGENCY if phase.get("emergency") else LANE_COLORS[phase["phase"] % len(LANE_COLORS)]
            cv2.rectangle(panel, (x1, row_y + 4), (x2, row_y + timeline_row - 4), color, -1)
    y += n * timeline_row + 8
    for t in range(0, int(sim_time) + 1, max(10, int(sim_time / 8 // 10 * 10) or 10)):
        tx = tl_x + int(t * scale)
        cv2.line(panel, (tx, y - 4), (tx, y), MUTED, 1)
        text(panel, "{}s".format(t), (tx - 8, y + 16), 0.38, MUTED)

    reasons = {}
    for phase in phases:
        if phase.get("end_reason"):
            reasons[phase["end_reason"]] = reasons.get(phase["end_reason"], 0) + 1
    reason_text = "Green ended by:  " + "   ".join("{} x{}".format(k, v) for k, v in sorted(reasons.items())) \
        if reasons else ""
    text(panel, reason_text, (32, height - 24), 0.45, MUTED)
    hint = "Press any key to close"
    text(panel, hint, (SUMMARY_WIDTH - text_size(hint, 0.45)[0] - 32, height - 24), 0.45, MUTED)
    return panel
