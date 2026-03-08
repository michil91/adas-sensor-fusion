"""
visualizer.py — Bird's-eye-view and 3D perspective plots of fused ADAS detections.

Both functions share the same visual language:
  - Green filled circles   → 'confirmed'   tracks
  - Orange triangles       → 'tentative'   tracks
  - Orange filled diamonds → 'cautionary'  tracks (single-sensor VRU, conf ≥ 0.60)
  - Red  x-marks           → 'unconfirmed' tracks
  - Marker area scales linearly with fused confidence
  - Dashed concentric rings mark the Near / Mid / Far fusion weight zones

When a weather condition is supplied, both plots gain:
  - A text box (upper-right) summarising sensor impacts for that condition
  - A dashed red circle at the LiDAR max-range cutoff (where applicable)
  - Heavy-rain-specific callout arrows for the safety-critical objects
    documented in the DESIGN NOTE in src/fusion.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")           # non-interactive backend; set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

from src.fusion import FusedObject


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("output")


# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

_BG      = "#0d1117"   # figure / outer background
_AX_BG   = "#161b22"   # axes face colour
_PANE_ED = "#30363d"   # 3-D pane edge colour

# Status → colour and matplotlib marker code.
_COLOUR = {
    "confirmed":   "#2ea043",   # green
    "tentative":   "#e69500",   # deep amber
    "cautionary":  "#fb923c",   # bright orange  (single-sensor VRU tier)
    "unconfirmed": "#f85149",   # red
}
_MARKER = {
    "confirmed":   "o",   # circle
    "tentative":   "^",   # triangle
    "cautionary":  "D",   # diamond
    "unconfirmed": "x",   # cross
}

# Zone boundary radii and their labels
_RINGS    = [50, 120, 200]
_RING_LBL = {50: "Near 50 m", 120: "Mid 120 m", 200: "200 m"}

EGO_COLOUR = "#58a6ff"


# ---------------------------------------------------------------------------
# Weather annotation constants
# ---------------------------------------------------------------------------

# Human-readable impact summary per condition (multi-line).
# None → no box rendered (clear weather).
_WEATHER_INFO: dict[str, str | None] = {
    "clear":      None,
    "rain":       (
        "Rain\n"
        "LiDAR conf: \u221220%  \u2022  Camera conf: \u22127.5%\n"
        "Radar conf: \u22125%"
    ),
    "heavy_rain": (
        "Heavy Rain\n"
        "LiDAR range limited to 80 m (rain scattering)\n"
        "Camera conf: \u221215%  \u2022  Radar conf: \u221210%"
    ),
    "fog":        (
        "Fog\n"
        "LiDAR range limited to 60 m\n"
        "Camera conf: \u221230%"
    ),
    "snow":       (
        "Snow\n"
        "LiDAR range limited to 100 m\n"
        "Camera conf: \u221220%  \u2022  Radar conf: \u221215%"
    ),
    "night":      "Night\nCamera conf: \u221225%",
    "glare":      "Glare\nCamera conf: \u221240% (forward 30\u00b0 cone)",
}

# LiDAR hard range cutoff (m) per condition.
# Conditions absent from this dict have no range cutoff.
_LIDAR_RANGE_CUTOFF: dict[str, float] = {
    "heavy_rain": 80.0,
    "fog":        60.0,
    "snow":       100.0,
}

# Callout text for the three heavy-rain safety findings.
# Unicode escapes: → U+2192, – U+2013, — U+2014
_CALLOUT_207 = (
    "ID 207 (pedestrian):\n"
    "confirmed \u2192 cautionary\n"
    "LiDAR lost to rain scattering at 95 m\n"
    "Camera-only detection triggers\n"
    "precautionary response"
)
_CALLOUT_208 = (
    "ID 208 (cyclist):\n"
    "confirmed \u2192 unconfirmed\n"
    "Camera conf 0.578 falls below\n"
    "0.60 cautionary threshold by 0.022"
)
_CALLOUT_108_109 = (
    "IDs 108\u2013109 (pedestrians):\n"
    "confirmed \u2192 tentative at 14\u201320 m\n"
    "Close-range confidence degradation\n"
    "\u2014 no sensor dropout"
)


# ---------------------------------------------------------------------------
# Internal helpers — shared
# ---------------------------------------------------------------------------

def _class_label(obj: FusedObject) -> str:
    """Return the best available class string for axis annotation."""
    if obj["camera"] is not None:
        return obj["camera"]["class_"]
    if obj["lidar"] is not None:
        return "pedestrian" if obj["lidar"]["z_m"] > 1.2 else "vehicle"
    return "obj"


def _timestamp_range(objects: list[FusedObject]) -> tuple[float, float]:
    """Return (min, max) Unix timestamps across all raw sensor detections."""
    stamps: list[float] = []
    for o in objects:
        for det in (o["camera"], o["radar"], o["lidar"]):
            if det is not None:
                stamps.append(det["timestamp"])
    return (min(stamps), max(stamps)) if stamps else (0.0, 0.0)


def _marker_size(confidence: float) -> float:
    """Map confidence ∈ [0, 1] → scatter marker area ∈ [50, 280]."""
    return 50.0 + 230.0 * confidence


def _legend_handles() -> list[Line2D]:
    """Build a fixed legend handle list with correct marker shapes."""
    entries = [
        ("Ego vehicle", EGO_COLOUR,              "D"),
        ("Confirmed",   _COLOUR["confirmed"],    _MARKER["confirmed"]),
        ("Tentative",   _COLOUR["tentative"],    _MARKER["tentative"]),
        ("Cautionary",  _COLOUR["cautionary"],   _MARKER["cautionary"]),
        ("Unconfirmed", _COLOUR["unconfirmed"],  _MARKER["unconfirmed"]),
    ]
    return [
        Line2D([0], [0], marker=mkr, color=col, label=lbl,
               markersize=8, linestyle="none",
               markeredgecolor="white" if mkr != "x" else col,
               markeredgewidth=0.5)
        for lbl, col, mkr in entries
    ]


def _by_id(objects: list[FusedObject]) -> dict[int, FusedObject]:
    return {o["object_id"]: o for o in objects}


# ---------------------------------------------------------------------------
# Internal helpers — bird's-eye-view weather annotations
# ---------------------------------------------------------------------------

def _bev_weather_box(ax: plt.Axes, weather_condition: str) -> None:
    """Draw the weather summary box in the upper-right corner."""
    text = _WEATHER_INFO.get(weather_condition)
    if not text:
        return
    ax.text(
        0.99, 0.99, text,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5, color="white", linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="#1c2128", edgecolor="#e69500", alpha=0.90),
        zorder=20,
    )


def _bev_lidar_ring(ax: plt.Axes, weather_condition: str) -> None:
    """Draw the LiDAR max-range dashed red circle on the BEV plot."""
    cutoff = _LIDAR_RANGE_CUTOFF.get(weather_condition)
    if cutoff is None:
        return
    theta = np.linspace(0.0, 2.0 * math.pi, 720)
    ax.plot(
        cutoff * np.cos(theta), cutoff * np.sin(theta),
        color="#f85149", alpha=0.75, linewidth=1.5, linestyle="--", zorder=4,
    )
    ax.text(
        cutoff + 1.5, 2.5,
        f"LiDAR cutoff {cutoff:.0f} m",
        color="#f85149", alpha=0.88, fontsize=7, va="bottom", zorder=4,
    )


def _bev_heavy_rain_callouts(ax: plt.Axes, objects: list[FusedObject]) -> None:
    """Add the three heavy-rain safety callout arrows to the BEV plot."""
    idx = _by_id(objects)

    _bev_callout = dict(
        color="white", fontsize=6.0, ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1c2128", alpha=0.88),
        zorder=15,
    )

    # --- Object 207: confirmed → cautionary ---
    o207 = idx.get(207)
    if o207:
        x, y = o207["fused_position"]["x_m"], o207["fused_position"]["y_m"]
        col = _COLOUR[o207["status"]]
        ax.annotate(
            _CALLOUT_207,
            xy=(x, y), xytext=(58, 30),
            va="bottom",
            arrowprops=dict(arrowstyle="->", color=col, lw=1.0,
                            connectionstyle="arc3,rad=0.18"),
            **{**_bev_callout,
               "bbox": {**_bev_callout["bbox"], "edgecolor": col}},
        )

    # --- Object 208: confirmed → unconfirmed ---
    o208 = idx.get(208)
    if o208:
        x, y = o208["fused_position"]["x_m"], o208["fused_position"]["y_m"]
        col = _COLOUR[o208["status"]]
        ax.annotate(
            _CALLOUT_208,
            xy=(x, y), xytext=(78, -31),
            va="top",
            arrowprops=dict(arrowstyle="->", color=col, lw=1.0,
                            connectionstyle="arc3,rad=-0.18"),
            **{**_bev_callout,
               "bbox": {**_bev_callout["bbox"], "edgecolor": col}},
        )

    # --- Objects 108/109: shared callout pointing to their midpoint ---
    pts = [(idx[oid]["fused_position"]["x_m"], idx[oid]["fused_position"]["y_m"])
           for oid in (108, 109) if oid in idx]
    if pts:
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        col = _COLOUR["tentative"]
        ax.annotate(
            _CALLOUT_108_109,
            xy=(mx, my), xytext=(4, 26),
            va="bottom",
            arrowprops=dict(arrowstyle="->", color=col, lw=1.0,
                            connectionstyle="arc3,rad=0.12"),
            **{**_bev_callout,
               "bbox": {**_bev_callout["bbox"], "edgecolor": col}},
        )


def _add_bev_weather(ax: plt.Axes, objects: list[FusedObject],
                     weather_condition: str) -> None:
    _bev_weather_box(ax, weather_condition)
    _bev_lidar_ring(ax, weather_condition)
    if weather_condition == "heavy_rain":
        _bev_heavy_rain_callouts(ax, objects)


# ---------------------------------------------------------------------------
# Internal helpers — 3-D perspective weather annotations
# ---------------------------------------------------------------------------

def _3d_weather_box(ax, weather_condition: str) -> None:
    """Draw the weather summary box in the upper-right corner of the 3D plot."""
    text = _WEATHER_INFO.get(weather_condition)
    if not text:
        return
    ax.text2D(
        0.99, 0.99, text,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5, color="white", linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="#1c2128", edgecolor="#e69500", alpha=0.90),
        zorder=20,
    )


def _3d_lidar_ring(ax, weather_condition: str) -> None:
    """Draw the LiDAR max-range red dashed circle on the ground plane."""
    cutoff = _LIDAR_RANGE_CUTOFF.get(weather_condition)
    if cutoff is None:
        return
    theta = np.linspace(0.0, 2.0 * math.pi, 720)
    ax.plot(
        cutoff * np.cos(theta), cutoff * np.sin(theta), np.zeros(720),
        color="#f85149", alpha=0.75, linewidth=1.5, linestyle="--", zorder=4,
    )


def _3d_stem_label(ax, x0: float, y0: float, z0: float,
                   xt: float, yt: float, zt: float,
                   text: str, col: str) -> None:
    """Draw a stem line from object to text-box position in 3-D space."""
    ax.plot([x0, xt], [y0, yt], [z0, zt],
            color=col, lw=0.9, alpha=0.80, zorder=10)
    ax.text(xt, yt, zt, text,
            color="white", fontsize=5.8, linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="#1c2128", edgecolor=col, alpha=0.88),
            zorder=11)


def _3d_heavy_rain_callouts(ax, objects: list[FusedObject]) -> None:
    """Add stem-line callout labels for the heavy-rain safety objects."""
    idx = _by_id(objects)

    def _z(o: FusedObject) -> float:
        return o["lidar"]["z_m"] if o["lidar"] is not None else 0.5

    # --- Object 207 ---
    o207 = idx.get(207)
    if o207:
        x, y = o207["fused_position"]["x_m"], o207["fused_position"]["y_m"]
        z = _z(o207)
        col = _COLOUR[o207["status"]]
        _3d_stem_label(ax, x, y, z, x - 26, y + 13, 1.75,
                       "ID 207 (pedestrian):\n"
                       "confirmed \u2192 cautionary\n"
                       "LiDAR lost at 95 m (rain)\n"
                       "Camera-only, precautionary", col)

    # --- Object 208 ---
    o208 = idx.get(208)
    if o208:
        x, y = o208["fused_position"]["x_m"], o208["fused_position"]["y_m"]
        z = _z(o208)
        col = _COLOUR[o208["status"]]
        _3d_stem_label(ax, x, y, z, x - 22, y - 17, 1.75,
                       "ID 208 (cyclist):\n"
                       "confirmed \u2192 unconfirmed\n"
                       "Camera conf 0.578 < 0.60\n"
                       "threshold (margin: 0.022)", col)

    # --- Objects 108 / 109 ---
    pts = [(idx[oid]["fused_position"]["x_m"],
            idx[oid]["fused_position"]["y_m"],
            _z(idx[oid]))
           for oid in (108, 109) if oid in idx]
    if pts:
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        mz = sum(p[2] for p in pts) / len(pts)
        col = _COLOUR["tentative"]
        _3d_stem_label(ax, mx, my, mz, mx - 18, my + 15, 2.1,
                       "IDs 108\u2013109 (pedestrians):\n"
                       "confirmed \u2192 tentative 14\u201320 m\n"
                       "Close-range conf degradation\n"
                       "\u2014 no sensor dropout", col)


def _add_3d_weather(ax, objects: list[FusedObject],
                    weather_condition: str) -> None:
    _3d_weather_box(ax, weather_condition)
    _3d_lidar_ring(ax, weather_condition)
    if weather_condition == "heavy_rain":
        _3d_heavy_rain_callouts(ax, objects)


# ---------------------------------------------------------------------------
# 2-D bird's-eye-view
# ---------------------------------------------------------------------------

def plot_birdseye(
    objects: list[FusedObject],
    output_path: Optional[Path | str] = None,
    weather_condition: Optional[str] = None,
) -> Path:
    """Render a 2-D top-down map of fused detections and save to PNG.

    Coordinate convention (matches the fusion module):
      x_m — longitudinal / forward  → plotted on the horizontal axis
      y_m — lateral / left positive → plotted on the vertical axis

    Ego vehicle is at the origin.  Dashed concentric rings mark the Near
    (50 m), Mid (120 m), and 200 m boundaries that correspond to the
    range-dependent fusion weight zones defined in ``src/fusion.py``.

    When *weather_condition* is provided (any value other than ``'clear'``),
    the plot gains a weather summary box, a LiDAR range-cutoff circle where
    applicable, and condition-specific object callouts.

    Parameters
    ----------
    objects:
        Fused track list from :func:`src.fusion.fuse_detections`.
    output_path:
        Destination file.  Defaults to ``output/fusion_result.png``.
    weather_condition:
        Optional weather condition string (e.g. ``'heavy_rain'``).  Controls
        which annotations are added.  ``None`` or ``'clear'`` → no weather
        annotations.

    Returns
    -------
    Path
        Resolved absolute path of the saved image.
    """
    out = Path(output_path or OUTPUT_DIR / "fusion_result.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_AX_BG)

    # ── Range rings ──────────────────────────────────────────────────────────
    theta = np.linspace(0.0, 2.0 * math.pi, 720)
    for r in _RINGS:
        ax.plot(
            r * np.cos(theta), r * np.sin(theta),
            color="white", alpha=0.08, linewidth=0.9, linestyle="--", zorder=1,
        )
        ax.text(
            r + 1.5, 2.0, _RING_LBL[r],
            color="white", alpha=0.30, fontsize=7, va="bottom",
        )

    # ── Ego vehicle ──────────────────────────────────────────────────────────
    ax.scatter(
        [0], [0], marker="D", color=EGO_COLOUR, s=140, zorder=10,
        edgecolors="white", linewidths=0.8,
    )
    ax.annotate(
        "EGO", xy=(0, 0), xytext=(4, 4), textcoords="offset points",
        color=EGO_COLOUR, fontsize=7, fontweight="bold",
    )

    # ── Fused objects ─────────────────────────────────────────────────────────
    for o in objects:
        x  = o["fused_position"]["x_m"]
        y  = o["fused_position"]["y_m"]
        st = o["status"]
        kw: dict = dict(
            marker=_MARKER[st], color=_COLOUR[st],
            s=_marker_size(o["fused_confidence"]),
            zorder=5, alpha=0.90, linewidths=0.5,
        )
        if _MARKER[st] != "x":
            kw["edgecolors"] = "white"
        ax.scatter([x], [y], **kw)
        ax.annotate(
            f"{o['object_id']} ({_class_label(o)})",
            xy=(x, y), xytext=(6, 5), textcoords="offset points",
            color="white", fontsize=6.5, alpha=0.92,
        )

    # ── Weather annotations (optional) ────────────────────────────────────────
    if weather_condition and weather_condition != "clear":
        _add_bev_weather(ax, objects, weather_condition)

    # ── Axes & styling ────────────────────────────────────────────────────────
    t_min, t_max = _timestamp_range(objects)
    weather_suffix = (
        f" — {weather_condition.replace('_', ' ').title()}"
        if weather_condition and weather_condition != "clear"
        else ""
    )
    ax.set_xlabel("x (m) — forward", color="#8b949e", fontsize=9, labelpad=6)
    ax.set_ylabel("y (m) — lateral (left +)", color="#8b949e", fontsize=9, labelpad=6)
    ax.set_title(
        f"ADAS Sensor Fusion — Bird's-Eye View{weather_suffix}\n"
        f"Timestamp  {t_min:.3f} – {t_max:.3f} s",
        color="white", fontsize=11, pad=12,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-25, 235)
    ax.set_ylim(-42, 42)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor(_PANE_ED)
    ax.grid(True, color=_PANE_ED, alpha=0.45, linewidth=0.4, zorder=0)

    ax.legend(
        handles=_legend_handles(),
        loc="upper left", framealpha=0.40, labelcolor="white",
        facecolor=_AX_BG, edgecolor=_PANE_ED, fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out.resolve()


# ---------------------------------------------------------------------------
# 3-D perspective view
# ---------------------------------------------------------------------------

def plot_3d(
    objects: list[FusedObject],
    output_path: Optional[Path | str] = None,
    weather_condition: Optional[str] = None,
) -> Path:
    """Render a 3-D perspective view of fused detections and save to PNG.

    Height data (``z_m``) is sourced from LiDAR cluster centroids where
    available; objects without a LiDAR detection are placed at z = 0.5 m
    (roughly centre-height of a small object on the road surface).

    Camera viewpoint
    ----------------
    ``ax.view_init(elev=25, azim=-110)`` approximates the scene coordinate
    position (−15, −10, 12) — 15 m behind the ego, 10 m to the right, 12 m
    above ground — looking toward the detection field (≈ 100 m ahead).

    Range rings are drawn as circles on the z = 0 ground plane.  When a
    weather condition with a LiDAR range cutoff is supplied, an additional
    dashed red ring marks the effective LiDAR boundary.

    Parameters
    ----------
    objects:
        Fused track list from :func:`src.fusion.fuse_detections`.
    output_path:
        Destination file.  Defaults to ``output/fusion_result_3d.png``.
    weather_condition:
        Optional weather condition string.  Controls which annotations are
        added.  ``None`` or ``'clear'`` → no weather annotations.

    Returns
    -------
    Path
        Resolved absolute path of the saved image.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers projection

    out = Path(output_path or OUTPUT_DIR / "fusion_result_3d.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(_BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(_AX_BG)

    # ── Subtle ground-plane grid ──────────────────────────────────────────────
    for xv in range(0, 231, 25):
        ax.plot([xv, xv], [-40.0, 40.0], [0.0, 0.0],
                color="white", alpha=0.06, linewidth=0.5, zorder=1)
    for yv in range(-40, 41, 20):
        ax.plot([0.0, 230.0], [float(yv), float(yv)], [0.0, 0.0],
                color="white", alpha=0.06, linewidth=0.5, zorder=1)

    # ── Range rings on z = 0 ─────────────────────────────────────────────────
    theta = np.linspace(0.0, 2.0 * math.pi, 720)
    for r in _RINGS:
        ax.plot(
            r * np.cos(theta), r * np.sin(theta), np.zeros(720),
            color="white", alpha=0.09, linewidth=0.9, linestyle="--", zorder=2,
        )

    # ── Ego vehicle ──────────────────────────────────────────────────────────
    ax.scatter(
        [0], [0], [0], marker="D", color=EGO_COLOUR, s=120, zorder=10,
        edgecolors="white", linewidths=0.8,
    )
    ax.text(0.0, 0.0, 0.45, "EGO", color=EGO_COLOUR, fontsize=7, fontweight="bold")

    # ── Fused objects ─────────────────────────────────────────────────────────
    for o in objects:
        x  = o["fused_position"]["x_m"]
        y  = o["fused_position"]["y_m"]
        z  = o["lidar"]["z_m"] if o["lidar"] is not None else 0.5
        st = o["status"]
        kw3: dict = dict(
            marker=_MARKER[st], color=_COLOUR[st],
            s=_marker_size(o["fused_confidence"]),
            alpha=0.90, zorder=5, linewidths=0.5,
        )
        if _MARKER[st] != "x":
            kw3["edgecolors"] = "white"
        ax.scatter([x], [y], [z], **kw3)
        ax.plot([x, x], [y, y], [0.0, z],
                color=_COLOUR[st], alpha=0.22, linewidth=0.6, zorder=3)
        ax.text(x, y, z + 0.40, str(o["object_id"]),
                color="white", fontsize=5.5, alpha=0.88)

    # ── Weather annotations (optional) ────────────────────────────────────────
    if weather_condition and weather_condition != "clear":
        _add_3d_weather(ax, objects, weather_condition)

    # ── Axes & styling ────────────────────────────────────────────────────────
    t_min, t_max = _timestamp_range(objects)
    weather_suffix = (
        f" — {weather_condition.replace('_', ' ').title()}"
        if weather_condition and weather_condition != "clear"
        else ""
    )
    ax.set_xlabel("x (m) — forward",  color="#8b949e", labelpad=8, fontsize=8)
    ax.set_ylabel("y (m) — lateral",  color="#8b949e", labelpad=8, fontsize=8)
    ax.set_zlabel("z (m) — height",   color="#8b949e", labelpad=8, fontsize=8)
    ax.set_title(
        f"ADAS Sensor Fusion — 3D Perspective{weather_suffix}\n"
        f"Timestamp  {t_min:.3f} – {t_max:.3f} s",
        color="white", fontsize=11, pad=15,
    )

    ax.set_xlim(0, 230)
    ax.set_ylim(-40, 40)
    ax.set_zlim(0, 2.5)   # raised slightly to accommodate annotation labels

    ax.tick_params(colors="#8b949e", labelsize=7)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(_PANE_ED)
    ax.grid(False)

    ax.view_init(elev=25, azim=-110)

    ax.legend(
        handles=_legend_handles(),
        loc="upper left", framealpha=0.40, labelcolor="white",
        facecolor=_AX_BG, edgecolor=_PANE_ED, fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out.resolve()
