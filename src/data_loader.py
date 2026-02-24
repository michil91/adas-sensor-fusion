"""
data_loader.py — Load and validate raw sensor detections from JSON files.

Sensor modality overview
------------------------
Camera
    A forward-facing monocular or stereo camera captures the scene at high
    frame rates (30–60 Hz) and produces rich semantic information: object
    class, colour, lane markings, traffic signs, and 2-D bounding boxes.
    Strengths : dense texture information, strong classification, low cost.
    Limitations: depth is ambiguous without stereo, performance degrades in
    low light, glare, rain, or fog, and the field of view is fixed.

Radar
    A frequency-modulated continuous-wave (FMCW) radar emits radio waves and
    measures the reflected signals to determine range, radial velocity
    (via the Doppler effect), and azimuth angle. Typical automotive radars
    operate at 77 GHz.
    Strengths : direct velocity measurement, works in all weather, long range
    (up to ~250 m), low latency.
    Limitations: poor angular resolution, limited height information, prone to
    multipath reflections and clutter from stationary infrastructure.

LiDAR
    A Light Detection And Ranging sensor spins one or more laser beams to
    build a 360° (or sector) 3-D point cloud at ~10–20 Hz. Each return gives
    a precise (x, y, z) position in the sensor frame.
    Strengths : accurate 3-D geometry, no ambiguity in range, large FOV.
    Limitations: performance degrades in heavy rain, snow, or fog (scattering),
    sparse returns at long range, high unit cost, large data volume.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TypedDicts — one per sensor modality
# ---------------------------------------------------------------------------

class BoundingBox(TypedDict):
    x: int   # left edge of box in pixels
    y: int   # top edge of box in pixels
    w: int   # width in pixels
    h: int   # height in pixels


class CameraDetection(TypedDict):
    object_id: int
    class_: str          # 'vehicle', 'pedestrian', or 'cyclist'
                         # stored as 'class' in JSON; renamed to avoid keyword clash
    bounding_box: BoundingBox
    confidence: float    # [0, 1] — detector posterior probability
    timestamp: float     # Unix epoch seconds (sub-second precision)


class RadarDetection(TypedDict):
    object_id: int
    range_m: float       # radial distance from sensor in metres
    velocity_mps: float  # radial velocity in m/s; negative = closing
    azimuth_deg: float   # horizontal angle from boresight in degrees
    confidence: float
    timestamp: float


class LidarDetection(TypedDict):
    object_id: int
    x_m: float           # longitudinal distance (forward) in metres
    y_m: float           # lateral offset (left positive) in metres
    z_m: float           # height above ground plane in metres
    num_points: int      # number of LiDAR returns belonging to this cluster
    confidence: float
    timestamp: float


class SensorData(TypedDict):
    camera: list[CameraDetection]
    radar: list[RadarDetection]
    lidar: list[LidarDetection]


# ---------------------------------------------------------------------------
# Required fields for each sensor type (used during validation)
# ---------------------------------------------------------------------------

_CAMERA_FIELDS: frozenset[str] = frozenset(
    {"object_id", "class", "bounding_box", "confidence", "timestamp"}
)
_BOUNDING_BOX_FIELDS: frozenset[str] = frozenset({"x", "y", "w", "h"})
_RADAR_FIELDS: frozenset[str] = frozenset(
    {"object_id", "range_m", "velocity_mps", "azimuth_deg", "confidence", "timestamp"}
)
_LIDAR_FIELDS: frozenset[str] = frozenset(
    {"object_id", "x_m", "y_m", "z_m", "num_points", "confidence", "timestamp"}
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> list[dict]:
    """Read and parse a JSON file, returning its contents as a list.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be decoded as JSON or its top-level value is not
        a list.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Sensor file not found: {path}")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array in {path}, got {type(data).__name__}"
        )

    return data


def _check_fields(record: dict, required: frozenset[str], label: str) -> list[str]:
    """Return a list of field names that are present in *required* but missing
    from *record*.  *label* is used only for log messages.
    """
    missing = required - record.keys()
    if missing:
        logger.warning(
            "%s object_id=%s is missing fields: %s",
            label,
            record.get("object_id", "<unknown>"),
            sorted(missing),
        )
    return list(missing)


# ---------------------------------------------------------------------------
# Per-sensor validators
# ---------------------------------------------------------------------------

def _validate_camera(records: list[dict]) -> list[CameraDetection]:
    """Validate and coerce a list of raw camera detection dicts.

    Each record must contain all fields in ``_CAMERA_FIELDS``.  The nested
    ``bounding_box`` dict is also checked for its four sub-fields.
    Records that fail validation are skipped with a warning.

    Parameters
    ----------
    records:
        Raw dicts as parsed from the JSON file.

    Returns
    -------
    list[CameraDetection]
        Valid detections with the ``class`` key remapped to ``class_`` so that
        it does not clash with the Python built-in.
    """
    validated: list[CameraDetection] = []
    for record in records:
        missing = _check_fields(record, _CAMERA_FIELDS, "camera")
        if missing:
            continue

        bbox = record["bounding_box"]
        if not isinstance(bbox, dict):
            logger.warning(
                "camera object_id=%s: bounding_box is not a dict, skipping",
                record.get("object_id"),
            )
            continue

        bbox_missing = _check_fields(bbox, _BOUNDING_BOX_FIELDS, "camera bbox")
        if bbox_missing:
            continue

        detection: CameraDetection = {
            "object_id": int(record["object_id"]),
            "class_": str(record["class"]),
            "bounding_box": {
                "x": int(bbox["x"]),
                "y": int(bbox["y"]),
                "w": int(bbox["w"]),
                "h": int(bbox["h"]),
            },
            "confidence": float(record["confidence"]),
            "timestamp": float(record["timestamp"]),
        }
        validated.append(detection)

    logger.info("camera: %d/%d records passed validation", len(validated), len(records))
    return validated


def _validate_radar(records: list[dict]) -> list[RadarDetection]:
    """Validate and coerce a list of raw radar detection dicts.

    Each record must contain all fields in ``_RADAR_FIELDS``.  Records that
    fail validation are skipped with a warning.

    Parameters
    ----------
    records:
        Raw dicts as parsed from the JSON file.

    Returns
    -------
    list[RadarDetection]
        Valid, type-coerced radar detections.
    """
    validated: list[RadarDetection] = []
    for record in records:
        missing = _check_fields(record, _RADAR_FIELDS, "radar")
        if missing:
            continue

        detection: RadarDetection = {
            "object_id": int(record["object_id"]),
            "range_m": float(record["range_m"]),
            "velocity_mps": float(record["velocity_mps"]),
            "azimuth_deg": float(record["azimuth_deg"]),
            "confidence": float(record["confidence"]),
            "timestamp": float(record["timestamp"]),
        }
        validated.append(detection)

    logger.info("radar: %d/%d records passed validation", len(validated), len(records))
    return validated


def _validate_lidar(records: list[dict]) -> list[LidarDetection]:
    """Validate and coerce a list of raw LiDAR detection dicts.

    Each record must contain all fields in ``_LIDAR_FIELDS``.  Records that
    fail validation are skipped with a warning.

    Parameters
    ----------
    records:
        Raw dicts as parsed from the JSON file.

    Returns
    -------
    list[LidarDetection]
        Valid, type-coerced LiDAR detections.
    """
    validated: list[LidarDetection] = []
    for record in records:
        missing = _check_fields(record, _LIDAR_FIELDS, "lidar")
        if missing:
            continue

        detection: LidarDetection = {
            "object_id": int(record["object_id"]),
            "x_m": float(record["x_m"]),
            "y_m": float(record["y_m"]),
            "z_m": float(record["z_m"]),
            "num_points": int(record["num_points"]),
            "confidence": float(record["confidence"]),
            "timestamp": float(record["timestamp"]),
        }
        validated.append(detection)

    logger.info("lidar: %d/%d records passed validation", len(validated), len(records))
    return validated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_sensor_data(data_dir: str | Path = "data") -> SensorData:
    """Load and validate all three sensor detection files from *data_dir*.

    Expected files
    --------------
    ``camera_detections.json``
        2-D bounding-box detections from the forward camera.
    ``radar_detections.json``
        Range, velocity, and azimuth measurements from the front radar.
    ``lidar_detections.json``
        3-D cluster centroids from the LiDAR point cloud.

    Parameters
    ----------
    data_dir:
        Path to the directory that contains the three JSON files.  Defaults
        to ``"data"`` (relative to the current working directory).

    Returns
    -------
    SensorData
        A typed dict with keys ``"camera"``, ``"radar"``, and ``"lidar"``,
        each mapping to a list of validated detection dicts.

    Raises
    ------
    FileNotFoundError
        If any of the three expected files is absent.
    ValueError
        If a file contains invalid JSON or its top-level value is not a list.

    Examples
    --------
    >>> from src.data_loader import load_sensor_data
    >>> data = load_sensor_data("data")
    >>> len(data["camera"])
    15
    >>> data["radar"][0]["range_m"]
    45.2
    """
    data_dir = Path(data_dir)

    camera_path = data_dir / "camera_detections.json"
    radar_path  = data_dir / "radar_detections.json"
    lidar_path  = data_dir / "lidar_detections.json"

    logger.info("Loading sensor data from %s", data_dir.resolve())

    raw_camera = _load_json(camera_path)
    raw_radar  = _load_json(radar_path)
    raw_lidar  = _load_json(lidar_path)

    return SensorData(
        camera=_validate_camera(raw_camera),
        radar=_validate_radar(raw_radar),
        lidar=_validate_lidar(raw_lidar),
    )
