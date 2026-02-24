"""
main.py — End-to-end ADAS sensor fusion pipeline.

Stages
------
1. Load      — read and validate all three sensor JSON files from data/
2. Fuse      — match detections, compute range-dependent weights, classify tracks
3. Report    — print a formatted summary table to the console
4. Visualise — save bird's-eye 2-D and 3-D perspective plots to output/
"""

import logging
import sys
from pathlib import Path

# Keep console output clean during normal runs; set DEBUG for detailed traces.
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s — %(message)s",
)

from src.data_loader import load_sensor_data
from src.fusion import fuse_detections, FusedObject, _representative_range, _sensor_weights
from src.visualizer import plot_birdseye, plot_3d


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_W   = 72
_SEP  = "═" * _W
_DASH = "─" * _W


def _class_label(o: FusedObject) -> str:
    if o["camera"] is not None:
        return o["camera"]["class_"]
    if o["lidar"] is not None:
        return "pedestrian" if o["lidar"]["z_m"] > 1.2 else "vehicle"
    return "—"


def _print_summary(objects: list[FusedObject]) -> None:
    confirmed   = [o for o in objects if o["status"] == "confirmed"]
    tentative   = [o for o in objects if o["status"] == "tentative"]
    cautionary  = [o for o in objects if o["status"] == "cautionary"]
    unconfirmed = [o for o in objects if o["status"] == "unconfirmed"]

    print(_DASH)
    print(
        f"  {'ID':>4}  {'status':<11}  {'class':<10}  {'sensors':<23}  "
        f"{'conf':>5}  {'x_m':>7}  {'y_m':>6}  {'rng':>6}  w L/R/C"
    )
    print(_DASH)

    for o in objects:
        r    = _representative_range(o["camera"], o["radar"], o["lidar"])
        w    = _sensor_weights(r)
        wstr = f"{w['lidar']:.2f}/{w['radar']:.2f}/{w['camera']:.2f}"
        sens = "+ ".join(o["sensors_detected_by"])
        print(
            f"  {o['object_id']:>4}  {o['status']:<11}  {_class_label(o):<10}  "
            f"{sens:<23}  {o['fused_confidence']:>5.3f}  "
            f"{o['fused_position']['x_m']:>7.1f}  {o['fused_position']['y_m']:>6.1f}  "
            f"{r:>5.1f}m  {wstr}"
        )

    print(_DASH)
    print(
        f"\n  Confirmed   ({len(confirmed):>2}):  "
        f"{[o['object_id'] for o in confirmed]}"
    )
    print(
        f"  Tentative   ({len(tentative):>2}):  "
        f"{[o['object_id'] for o in tentative]}"
    )
    print(
        f"  Cautionary  ({len(cautionary):>2}):  "
        f"{[o['object_id'] for o in cautionary]}"
    )
    print(
        f"  Unconfirmed ({len(unconfirmed):>2}):  "
        f"{[o['object_id'] for o in unconfirmed]}"
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    print(_SEP)
    print("  ADAS Sensor Fusion Pipeline")
    print(_SEP)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n[1/4]  Loading sensor data from data/")
    try:
        data = load_sensor_data("data")
    except (FileNotFoundError, ValueError) as exc:
        print(f"       ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    n_cam = len(data["camera"])
    n_rad = len(data["radar"])
    n_lid = len(data["lidar"])
    print(f"       camera  {n_cam:>2} detections")
    print(f"       radar   {n_rad:>2} detections")
    print(f"       lidar   {n_lid:>2} detections")
    print(f"       total   {n_cam + n_rad + n_lid:>2} raw detections across 3 sensors")

    # ── 2. Fuse ───────────────────────────────────────────────────────────────
    print("\n[2/4]  Running sensor fusion...")
    objects = fuse_detections(data)
    n_conf  = sum(1 for o in objects if o["status"] == "confirmed")
    n_tent  = sum(1 for o in objects if o["status"] == "tentative")
    n_caut  = sum(1 for o in objects if o["status"] == "cautionary")
    n_unco  = sum(1 for o in objects if o["status"] == "unconfirmed")
    print(f"       {len(objects)} fused tracks — "
          f"{n_conf} confirmed, {n_tent} tentative, "
          f"{n_caut} cautionary, {n_unco} unconfirmed")

    # ── 3. Report ─────────────────────────────────────────────────────────────
    print(f"\n[3/4]  Fused object list\n")
    _print_summary(objects)

    # ── 4. Visualise ──────────────────────────────────────────────────────────
    print(f"\n[4/4]  Generating visualizations...")
    path_2d = plot_birdseye(objects)
    print(f"       Bird's-eye view  →  {path_2d}")
    path_3d = plot_3d(objects)
    print(f"       3D perspective   →  {path_3d}")

    print(f"\n{_SEP}")
    print("  Pipeline complete.")
    print(_SEP)


if __name__ == "__main__":
    main()
