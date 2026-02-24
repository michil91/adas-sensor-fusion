"""
visualizer.py — Bird's-eye-view and 3D perspective plots of fused ADAS detections.

Both functions share the same visual language:
  - Green filled circles   → 'confirmed'   tracks
  - Orange triangles       → 'tentative'   tracks
  - Orange filled diamonds → 'cautionary'  tracks (single-sensor VRU, conf ≥ 0.60)
  - Red  x-marks           → 'unconfirmed' tracks
  - Marker area scales linearly with fused confidence
  - Dashed concentric rings mark the Near / Mid / Far fusion weight zones
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
# 'cautionary' uses a brighter orange-amber distinct from tentative's deeper
# amber, with a diamond shape to differentiate from tentative's triangle.
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

# Zone boundary radii and their names
_RINGS    = [50, 120, 200]
_RING_LBL = {50: "Near 50 m", 120: "Mid 120 m", 200: "200 m"}

EGO_COLOUR = "#58a6ff"


# ---------------------------------------------------------------------------
# Internal helpers
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


# ---------------------------------------------------------------------------
# 2-D bird's-eye-view
# ---------------------------------------------------------------------------

def plot_birdseye(
    objects: list[FusedObject],
    output_path: Optional[Path | str] = None,
) -> Path:
    """Render a 2-D top-down map of fused detections and save to PNG.

    Coordinate convention (matches the fusion module):
      x_m — longitudinal / forward  → plotted on the horizontal axis
      y_m — lateral / left positive → plotted on the vertical axis

    Ego vehicle is at the origin.  Dashed concentric rings mark the Near
    (50 m), Mid (120 m), and 200 m boundaries that correspond to the
    range-dependent fusion weight zones defined in ``src/fusion.py``.

    Parameters
    ----------
    objects:
        Fused track list from :func:`src.fusion.fuse_detections`.
    output_path:
        Destination file.  Defaults to ``output/fusion_result.png``.

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
        # 'x' is an unfilled marker — passing edgecolors triggers a UserWarning
        # in matplotlib ≤3.7 regardless of value, so omit it for 'x' markers.
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

    # ── Axes & styling ────────────────────────────────────────────────────────
    t_min, t_max = _timestamp_range(objects)
    ax.set_xlabel("x (m) — forward", color="#8b949e", fontsize=9, labelpad=6)
    ax.set_ylabel("y (m) — lateral (left +)", color="#8b949e", fontsize=9, labelpad=6)
    ax.set_title(
        "ADAS Sensor Fusion — Bird's-Eye View\n"
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
    The elevation is raised slightly from the geometric 6° to give a cleaner
    separation of near and far objects in the rendered image.

    Range rings are drawn as circles on the z = 0 ground plane so they remain
    visible from any camera elevation.

    Parameters
    ----------
    objects:
        Fused track list from :func:`src.fusion.fuse_detections`.
    output_path:
        Destination file.  Defaults to ``output/fusion_result_3d.png``.

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
        # Vertical stem from object down to ground plane
        ax.plot([x, x], [y, y], [0.0, z],
                color=_COLOUR[st], alpha=0.22, linewidth=0.6, zorder=3)
        ax.text(x, y, z + 0.40, str(o["object_id"]),
                color="white", fontsize=5.5, alpha=0.88)

    # ── Axes & styling ────────────────────────────────────────────────────────
    t_min, t_max = _timestamp_range(objects)
    ax.set_xlabel("x (m) — forward",  color="#8b949e", labelpad=8, fontsize=8)
    ax.set_ylabel("y (m) — lateral",  color="#8b949e", labelpad=8, fontsize=8)
    ax.set_zlabel("z (m) — height",   color="#8b949e", labelpad=8, fontsize=8)
    ax.set_title(
        "ADAS Sensor Fusion — 3D Perspective\n"
        f"Timestamp  {t_min:.3f} – {t_max:.3f} s",
        color="white", fontsize=11, pad=15,
    )

    ax.set_xlim(0, 230)
    ax.set_ylim(-40, 40)
    ax.set_zlim(0, 2)

    ax.tick_params(colors="#8b949e", labelsize=7)

    # Transparent axis panes with subtle edge colour
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(_PANE_ED)
    ax.grid(False)

    # Camera viewpoint: approximates position (−15, −10, 12) looking toward
    # the detection field centre (~100 m ahead of ego).
    # elev=25°  — vertical look-down angle (raised from geometric ~6° for clarity)
    # azim=−110° — positions camera on the −x side, rotated slightly toward −y
    #              (right of vehicle in our left-positive-y convention)
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
