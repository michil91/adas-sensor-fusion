"""
environment.py — Weather/environment simulation for ADAS sensor degradation.

Simulates how different weather conditions degrade the confidence and range
of each sensor modality (LiDAR, radar, camera) before fusion.
"""

from __future__ import annotations

import copy
import math
from enum import Enum
from typing import TypedDict

from src.data_loader import SensorData, CameraDetection, RadarDetection, LidarDetection


class WeatherCondition(Enum):
    CLEAR      = "clear"
    RAIN       = "rain"
    HEAVY_RAIN = "heavy_rain"
    FOG        = "fog"
    SNOW       = "snow"
    GLARE      = "glare"
    NIGHT      = "night"


class DegradationReport(TypedDict):
    condition:               str
    lidar_dropped:           int
    lidar_avg_conf_reduction:  float
    radar_dropped:           int
    radar_avg_conf_reduction:  float
    camera_dropped:          int
    camera_avg_conf_reduction: float


# Camera geometry constants used to estimate azimuth from bounding-box pixel x.
# Assumes a typical 1920-wide sensor with a 90° horizontal field of view.
_CAMERA_IMAGE_WIDTH = 1920
_CAMERA_HFOV_DEG    = 90.0
_GLARE_HALF_CONE    = 15.0   # ±15° → 30° forward cone


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _lidar_range(det: LidarDetection) -> float:
    """2-D ground-plane range from the sensor origin."""
    return math.hypot(det["x_m"], det["y_m"])


def _camera_azimuth_deg(det: CameraDetection) -> float:
    """Estimate azimuth in degrees from bounding-box centre x pixel.

    Maps pixel x ∈ [0, _CAMERA_IMAGE_WIDTH] linearly to
    [-HFOV/2, +HFOV/2] degrees.  Negative = left of boresight.
    """
    cx = det["bounding_box"]["x"] + det["bounding_box"]["w"] / 2.0
    image_centre = _CAMERA_IMAGE_WIDTH / 2.0
    return (cx - image_centre) / image_centre * (_CAMERA_HFOV_DEG / 2.0)


# ---------------------------------------------------------------------------
# Per-sensor degradation helpers
# ---------------------------------------------------------------------------

def _degrade_lidar(
    detections: list[LidarDetection],
    conf_factor: float,
    max_range_m: float | None,
) -> tuple[list[LidarDetection], int, float]:
    """Apply confidence scaling and optional range cutoff to LiDAR detections.

    Returns
    -------
    (kept, n_dropped, avg_conf_reduction)
    """
    kept: list[LidarDetection] = []
    total_reduction = 0.0
    dropped = 0

    for det in detections:
        if max_range_m is not None and _lidar_range(det) > max_range_m:
            dropped += 1
            continue
        original = det["confidence"]
        new_conf = max(0.0, original * conf_factor)
        total_reduction += original - new_conf
        kept.append({**det, "confidence": new_conf})

    n_total = len(detections)
    avg_reduction = total_reduction / n_total if n_total else 0.0
    return kept, dropped, avg_reduction


def _degrade_radar(
    detections: list[RadarDetection],
    conf_factor: float,
) -> tuple[list[RadarDetection], float]:
    """Scale radar detection confidences.

    Returns
    -------
    (modified_detections, avg_conf_reduction)
    """
    result: list[RadarDetection] = []
    total_reduction = 0.0

    for det in detections:
        original = det["confidence"]
        new_conf = max(0.0, original * conf_factor)
        total_reduction += original - new_conf
        result.append({**det, "confidence": new_conf})

    avg_reduction = total_reduction / len(detections) if detections else 0.0
    return result, avg_reduction


def _degrade_camera(
    detections: list[CameraDetection],
    conf_factor: float,
    glare_cone_only: bool = False,
) -> tuple[list[CameraDetection], float]:
    """Scale camera detection confidences, optionally restricted to the glare cone.

    When *glare_cone_only* is True only detections whose estimated azimuth
    falls within ±``_GLARE_HALF_CONE`` degrees are degraded.

    Returns
    -------
    (modified_detections, avg_conf_reduction)
        avg_conf_reduction is averaged over **all** detections (including those
        not degraded), so it reflects the mean impact across the full list.
    """
    result: list[CameraDetection] = []
    total_reduction = 0.0

    for det in detections:
        original = det["confidence"]
        if glare_cone_only and abs(_camera_azimuth_deg(det)) > _GLARE_HALF_CONE:
            result.append(det)
            continue
        new_conf = max(0.0, original * conf_factor)
        total_reduction += original - new_conf
        result.append({**det, "confidence": new_conf})

    avg_reduction = total_reduction / len(detections) if detections else 0.0
    return result, avg_reduction


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def degrade_sensor_data(
    data: SensorData,
    condition: WeatherCondition,
) -> tuple[SensorData, DegradationReport]:
    """Apply weather-induced sensor degradation to a copy of *data*.

    The original *data* is never mutated — all modifications are applied to
    deep copies.

    Degradation rules
    -----------------
    clear      : no changes.
    rain       : half the heavy_rain confidence penalties; no range cutoff.
    heavy_rain : LiDAR −40 % conf, drop > 80 m; radar −10 % conf;
                 camera −15 % conf.
    fog        : LiDAR −50 % conf, drop > 60 m; radar unaffected;
                 camera −30 % conf.
    snow       : LiDAR −30 % conf, drop > 100 m; radar −15 % conf;
                 camera −20 % conf.
    night      : LiDAR unaffected; radar unaffected; camera −25 % conf.
    glare      : LiDAR unaffected; radar unaffected; camera −40 % conf
                 for objects in the forward 30-degree cone only.

    Parameters
    ----------
    data:
        Validated sensor data as returned by ``load_sensor_data()``.
    condition:
        The weather/environment scenario to simulate.

    Returns
    -------
    degraded_data:
        A new ``SensorData`` dict with modified detections.
    report:
        ``DegradationReport`` summarising detections dropped and average
        confidence reduction per sensor.
    """
    lidar  = copy.deepcopy(data["lidar"])
    radar  = copy.deepcopy(data["radar"])
    camera = copy.deepcopy(data["camera"])

    lidar_dropped  = 0
    radar_dropped  = 0
    camera_dropped = 0
    lidar_avg_red  = 0.0
    radar_avg_red  = 0.0
    camera_avg_red = 0.0

    if condition == WeatherCondition.CLEAR:
        pass

    elif condition == WeatherCondition.HEAVY_RAIN:
        lidar,  lidar_dropped, lidar_avg_red  = _degrade_lidar(lidar,  0.60, 80.0)
        radar,                 radar_avg_red  = _degrade_radar(radar,  0.90)
        camera,                camera_avg_red = _degrade_camera(camera, 0.85)

    elif condition == WeatherCondition.RAIN:
        # Half the heavy_rain confidence penalties; no hard range cutoff.
        lidar,  lidar_dropped, lidar_avg_red  = _degrade_lidar(lidar,  0.80, None)
        radar,                 radar_avg_red  = _degrade_radar(radar,  0.95)
        camera,                camera_avg_red = _degrade_camera(camera, 0.925)

    elif condition == WeatherCondition.FOG:
        lidar,  lidar_dropped, lidar_avg_red  = _degrade_lidar(lidar,  0.50, 60.0)
        # radar: unaffected
        camera,                camera_avg_red = _degrade_camera(camera, 0.70)

    elif condition == WeatherCondition.SNOW:
        lidar,  lidar_dropped, lidar_avg_red  = _degrade_lidar(lidar,  0.70, 100.0)
        radar,                 radar_avg_red  = _degrade_radar(radar,  0.85)
        camera,                camera_avg_red = _degrade_camera(camera, 0.80)

    elif condition == WeatherCondition.NIGHT:
        # lidar: unaffected, radar: unaffected
        camera,                camera_avg_red = _degrade_camera(camera, 0.75)

    elif condition == WeatherCondition.GLARE:
        # lidar: unaffected, radar: unaffected
        # camera: −40 % only for objects inside the forward 30° cone
        camera,                camera_avg_red = _degrade_camera(camera, 0.60, glare_cone_only=True)

    degraded: SensorData = SensorData(camera=camera, radar=radar, lidar=lidar)

    report: DegradationReport = {
        "condition":               condition.value,
        "lidar_dropped":           lidar_dropped,
        "lidar_avg_conf_reduction":  lidar_avg_red,
        "radar_dropped":           radar_dropped,
        "radar_avg_conf_reduction":  radar_avg_red,
        "camera_dropped":          camera_dropped,
        "camera_avg_conf_reduction": camera_avg_red,
    }

    return degraded, report
