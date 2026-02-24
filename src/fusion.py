"""
fusion.py — Multi-sensor fusion pipeline for ADAS object detection.

Pipeline overview
-----------------
Raw detections from camera, radar, and LiDAR are matched by object_id (a
stand-in for a real data-association step) and fused into a single list of
FusedObject tracks.  Each track carries:

  * A fused position in the ego-vehicle ground-plane frame (x forward, y left).
  * A fused confidence score computed as a weighted average across sensors.
  * A status label — 'confirmed', 'tentative', 'cautionary', or 'unconfirmed'
    — based on how many sensors corroborate the detection, how confident they
    are, and whether the detected object is a Vulnerable Road User (VRU).

Coordinate frame
----------------
All positions are expressed in the ego-vehicle frame:
  x_m  — longitudinal distance ahead of the ego vehicle (metres)
  y_m  — lateral offset, positive to the left (metres)

Sensor weights (range-dependent rationale)
------------------------------------------
Weights are not fixed scalars — they vary continuously with object range
because each modality's reliability degrades at a different rate with distance:

  LiDAR  — point density falls as 1/r² (inverse-square law).  At close range
            (<50 m) a vehicle returns 100–200 points, giving a well-resolved
            centroid with sub-centimetre accuracy.  Beyond ~100 m the same
            vehicle may return fewer than 20 points; the centroid becomes
            noisy and the cluster can fragment or merge with neighbours.
            LiDAR therefore carries the highest near-range weight and the
            lowest far-range weight.

  Radar  — time-of-flight range measurement is independent of distance: 77 GHz
            FMCW radar maintains millimetre-level range accuracy out to 250 m
            regardless of target size.  Lateral (azimuth) resolution is still
            coarser than LiDAR at all ranges, but that limitation matters less
            at long range because even LiDAR's lateral accuracy degrades there.
            Radar therefore increases in weight as range grows, becoming the
            dominant modality in the far zone.

  Camera — classification confidence (detecting that an object *is* a vehicle
            rather than clutter) is largely stable with range: the detector
            sees a smaller but higher-contrast object and CNN features remain
            discriminative.  Monocular depth accuracy is poor at all ranges, so
            the camera's position weight stays low.  Its confidence contribution
            rises slightly at far range to reflect that it may still confirm an
            object class even when LiDAR geometry is sparse.

Zone anchor points (r = 0 m, 50 m, 120 m):

    Zone        Range       LiDAR   Radar   Camera
    ─────────────────────────────────────────────────
    Near        0 – 50 m    0.50    0.25    0.25
    Mid        50 – 120 m   0.35    0.40    0.25
    Far        120 m +      0.15    0.55    0.30

Between anchor points, weights are linearly interpolated.  Because every
anchor set sums to 1.0, the interpolated weights also sum to 1.0 at every
range without an explicit renormalisation step.

Real-world note: a production system would replace these piecewise-linear
curves with per-sensor covariance models inside an Extended Kalman Filter,
giving each measurement a weight proportional to 1/σ² dynamically computed
from target range, point count, SNR, and weather conditions.

Track classification and safety rationale
-----------------------------------------
Detections are assigned one of four status tiers:

  confirmed   — corroborated by ≥ 2 sensors with fused confidence above the
                class-appropriate threshold.  Eligible for safety-critical
                responses such as Automatic Emergency Braking (AEB).

  tentative   — corroborated by ≥ 2 sensors but below the confidence
                threshold.  Worth tracking but not yet actionable.

  cautionary  — single sensor only, but the detected class is a Vulnerable
                Road User (pedestrian or cyclist) and confidence ≥ 0.60.
                In production ADAS, this tier would trigger a precautionary
                speed reduction and driver alert rather than full AEB
                activation — a conservative response commensurate with the
                evidence quality.

  unconfirmed — single sensor, non-VRU class, or VRU with confidence < 0.60.
                Logged for transparency but not acted upon.

Why class-dependent thresholds?

  The asymmetric risk of a missed VRU detection versus a false positive
  justifies lower confirmation thresholds for pedestrians and cyclists.
  A missed vehicle detection at low speed has a different consequence profile
  than a missed pedestrian detection at the same speed.  Lowering the VRU
  threshold from 0.70 to 0.50 accepts a higher false-positive rate on that
  class in exchange for improved recall — the same risk-asymmetry logic that
  motivates distinct pedestrian AEB requirements in Euro NCAP.

  These thresholds are engineering design decisions justified by risk
  asymmetry analysis, not values prescribed by any regulatory standard.
  ISO 26262 defines process rigour and functional safety requirements;
  Euro NCAP defines system-level performance metrics.  Neither standard
  specifies internal fusion thresholds — those are left to the implementer's
  Hazard Analysis and Risk Assessment (HARA).
"""

from __future__ import annotations

import math
import logging
from typing import Optional, TypedDict

from src.data_loader import (
    CameraDetection,
    LidarDetection,
    RadarDetection,
    SensorData,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Range-dependent sensor weight anchors
# ---------------------------------------------------------------------------

# Breakpoints (metres) at which the weight profiles are anchored.
# Interpolation is piecewise-linear between consecutive breakpoints and
# clamped (flat) outside the [_NEAR_M, _FAR_M] interval.
_NEAR_M: float =   0.0
_MID_M:  float =  50.0
_FAR_M:  float = 120.0

# Anchor weight dicts — one per breakpoint.  Each set sums to exactly 1.0,
# which guarantees that any linear combination of two anchor sets also sums
# to 1.0 (no renormalisation needed after interpolation).
_W_NEAR: dict[str, float] = {"lidar": 0.50, "radar": 0.25, "camera": 0.25}
_W_MID:  dict[str, float] = {"lidar": 0.35, "radar": 0.40, "camera": 0.25}
_W_FAR:  dict[str, float] = {"lidar": 0.15, "radar": 0.55, "camera": 0.30}

# ---------------------------------------------------------------------------
# Camera intrinsic / extrinsic constants
# ---------------------------------------------------------------------------

# Assumed intrinsics for a 1920×1080 forward-facing ADAS camera.
# A narrow FoV (~27° HFOV) is typical for long-range highway detection; the
# corresponding focal length is f = (W/2) / tan(HFOV/2) ≈ 3900–4200 px.
# We use a round value that is broadly consistent with the synthetic bounding
# boxes in this dataset.
_FOCAL_LENGTH_PX: float = 4000.0

_IMAGE_WIDTH:  int = 1920
_IMAGE_HEIGHT: int = 1080
_CX: float = _IMAGE_WIDTH  / 2.0   # principal point x = 960.0 px
_CY: float = _IMAGE_HEIGHT / 2.0   # principal point y = 540.0 px

# Camera mounting height above the road surface.  Used only in comments; the
# size-based projection below does not require it explicitly.
_CAMERA_HEIGHT_M: float = 1.35     # metres, typical for a passenger sedan

# Canonical real-world widths per class for size-based depth estimation.
# These are median values; individual instances vary (compact car ~1.7 m,
# SUV ~2.1 m).  The ±15 % spread in real object width maps directly to
# ±15 % range error, which is why camera carries the lowest position weight.
_CLASS_WIDTH_M: dict[str, float] = {
    "vehicle":    2.0,
    "pedestrian": 0.5,
    "cyclist":    0.7,
}
_DEFAULT_WIDTH_M: float = 1.5  # fallback for unknown classes

# ---------------------------------------------------------------------------
# Track classification thresholds
# ---------------------------------------------------------------------------

# Vulnerable Road User classes.  These receive a lower confirmation threshold
# and a dedicated 'cautionary' single-sensor tier because the cost of a missed
# detection far exceeds the cost of a false positive.  See the module docstring
# for the full safety rationale.
_VRU_CLASSES: frozenset[str] = frozenset({"pedestrian", "cyclist"})

# Multi-sensor confirmation thresholds (fused confidence must reach or exceed
# this value for a multi-sensor track to be classified as 'confirmed').
_VRU_CONFIRMED_THRESHOLD:     float = 0.50   # lower bar: prioritise VRU recall
_NON_VRU_CONFIRMED_THRESHOLD: float = 0.70   # standard bar for vehicles etc.

# Single-sensor VRU detections at or above this confidence become 'cautionary'
# rather than 'unconfirmed', warranting a precautionary vehicle response.
_CAUTIONARY_THRESHOLD: float = 0.60


# ---------------------------------------------------------------------------
# Output TypedDicts
# ---------------------------------------------------------------------------

class FusedPosition(TypedDict):
    x_m: float   # longitudinal (forward) distance in metres
    y_m: float   # lateral offset in metres (positive = left of ego)


class FusedObject(TypedDict):
    object_id: int
    status: str                          # 'confirmed', 'tentative', 'unconfirmed'
    sensors_detected_by: list[str]       # e.g. ['lidar', 'radar', 'camera']
    fused_confidence: float              # weighted-average confidence in [0, 1]
    fused_position: FusedPosition        # estimated ground-plane location
    camera: Optional[CameraDetection]    # raw camera detection, or None
    radar:  Optional[RadarDetection]     # raw radar  detection, or None
    lidar:  Optional[LidarDetection]     # raw lidar  detection, or None


# ---------------------------------------------------------------------------
# Coordinate conversion — one function per sensor
# ---------------------------------------------------------------------------

def _camera_to_xy(det: CameraDetection) -> tuple[float, float]:
    """Estimate the ground-plane (x, y) position from a camera bounding box.

    Method — size-based monocular depth (thin-lens equation)
    ---------------------------------------------------------
    Given the known real-world width *W* of an object class and its apparent
    width *w_px* in pixels, depth (forward distance) is:

        x_m = focal_length × W / w_px

    Lateral offset is then recovered from the horizontal displacement of the
    bounding-box centre from the principal point:

        y_m = (u_centre − cx) × x_m / focal_length

    Limitations
    -----------
    * Range error scales linearly with width uncertainty.  A cyclist that is
      narrower than the 0.7 m canonical value will appear further away.
    * Partially occluded objects have artificially small bounding boxes,
      causing the range to be over-estimated.
    * This approach breaks down for objects very close to the camera (< 5 m)
      where the bounding box may extend beyond the image boundary.

    These limitations motivate the low camera weight (0.25) in position fusion.
    """
    bbox = det["bounding_box"]

    if bbox["w"] <= 0:
        # Degenerate box — cannot project; signal with NaN so fuse_position
        # can drop this measurement gracefully.
        return (float("nan"), float("nan"))

    real_width = _CLASS_WIDTH_M.get(det["class_"], _DEFAULT_WIDTH_M)

    # Forward distance via thin-lens equation.
    x_m = _FOCAL_LENGTH_PX * real_width / bbox["w"]

    # Lateral offset: positive = left of image centre = left of ego vehicle.
    u_centre = bbox["x"] + bbox["w"] / 2.0
    y_m = (u_centre - _CX) * x_m / _FOCAL_LENGTH_PX

    return (x_m, y_m)


def _radar_to_xy(det: RadarDetection) -> tuple[float, float]:
    """Convert radar polar coordinates to Cartesian (x, y).

    Radar reports measurements in polar form: range *r* and azimuth *θ*.
    We assume the radar boresight is co-aligned with the ego-vehicle's
    longitudinal axis (0° = straight ahead), with positive azimuth to the
    left (right-hand rule, z up):

        x_m = r · cos(θ)    — longitudinal (forward) distance
        y_m = r · sin(θ)    — lateral offset (positive left)

    Range accuracy is excellent (< 0.1 m RMS for 77 GHz FMCW).  Azimuth
    resolution is coarser — typically 1–3° for a single-beam radar — which
    translates to lateral uncertainty of roughly r · sin(Δθ) ≈ 0.9–2.6 m
    at 50 m range.  That is why radar carries a lower position weight than
    LiDAR despite its superior range accuracy.
    """
    theta = math.radians(det["azimuth_deg"])
    x_m = det["range_m"] * math.cos(theta)
    y_m = det["range_m"] * math.sin(theta)
    return (x_m, y_m)


def _lidar_to_xy(det: LidarDetection) -> tuple[float, float]:
    """Return the LiDAR cluster centroid projected onto the ground plane.

    LiDAR detections are already expressed in the ego-vehicle Cartesian frame
    (x forward, y left, z up), so the horizontal components are used directly.
    The z coordinate is discarded for 2-D fusion but is preserved in the raw
    detection for downstream 3-D tasks (height filtering, bridge detection).

    LiDAR position accuracy is typically < 0.05 m RMS at close range,
    degrading to ~0.1–0.2 m beyond 100 m as point density drops.  The
    num_points field is a quality indicator: clusters with ≥ 50 points are
    well-resolved, while single-digit clusters (like object 401 in the
    synthetic dataset) are marginal.
    """
    return (det["x_m"], det["y_m"])


# ---------------------------------------------------------------------------
# Range-dependent weight functions
# ---------------------------------------------------------------------------

# Why a range-dependent function rather than fixed scalars?
#
# Each sensor modality has a characteristic reliability curve with distance:
#   - LiDAR point density falls as 1/r², so its position weight should taper
#     off rapidly beyond ~50 m where clusters become sparse.
#   - Radar time-of-flight accuracy is distance-independent, making it
#     increasingly valuable at long range relative to the other sensors.
#   - Camera classification quality is broadly stable with range, so its
#     weight rises slightly at far range to reflect that class confirmation
#     remains useful even when geometric sensors are struggling.
#
# Encoding this as a piecewise-linear interpolation is the simplest model
# that captures these trends without introducing free parameters that are
# difficult to tune without a real sensor calibration dataset.

def _representative_range(
    cam: Optional[CameraDetection],
    rad: Optional[RadarDetection],
    lid: Optional[LidarDetection],
) -> float:
    """Return the best available range estimate for weight-profile selection.

    Priority order (most accurate first):

    1. LiDAR ``x_m``     — direct 3-D centroid; sub-centimetre accuracy.
    2. Radar ``range_m`` — time-of-flight; millimetre-level accuracy.
    3. Camera depth      — size-based thin-lens estimate; ±20–40 % accuracy.

    The returned value is used only to select the interpolated weight set and
    is not stored in the fused output.
    """
    if lid is not None:
        return lid["x_m"]
    if rad is not None:
        return rad["range_m"]
    if cam is not None and cam["bounding_box"]["w"] > 0:
        real_width = _CLASS_WIDTH_M.get(cam["class_"], _DEFAULT_WIDTH_M)
        return _FOCAL_LENGTH_PX * real_width / cam["bounding_box"]["w"]
    return _MID_M  # fallback: mid-range assumption


def _sensor_weights(range_m: float) -> dict[str, float]:
    """Return linearly interpolated sensor weights for a given range.

    Piecewise-linear interpolation between three anchor points:

        r ≤ 0 m    →  W_NEAR  (clamped at near anchor)
        r = 50 m   →  W_MID
        r = 120 m  →  W_FAR
        r > 120 m  →  W_FAR   (clamped at far anchor)

    Because every anchor weight set sums to 1.0, the interpolated result
    also sums to 1.0 at every range.  When a sensor is absent for a
    particular object its weight is simply omitted; ``_weighted_average``
    normalises the remaining weights, preserving the relative ratios.

    Parameters
    ----------
    range_m:
        Forward distance to the object in metres.  Negative values are
        treated as 0 (clamped to the near anchor).

    Returns
    -------
    dict[str, float]
        Keys ``'lidar'``, ``'radar'``, ``'camera'`` summing to 1.0.

    Examples
    --------
    >>> _sensor_weights(0)
    {'lidar': 0.50, 'radar': 0.25, 'camera': 0.25}
    >>> _sensor_weights(120)
    {'lidar': 0.15, 'radar': 0.55, 'camera': 0.30}
    >>> w = _sensor_weights(85)   # midpoint of mid zone (t=0.5)
    >>> round(sum(w.values()), 10)
    1.0
    """
    if range_m <= _MID_M:
        # Blend from W_NEAR (r=0) to W_MID (r=50).
        t = max(0.0, range_m) / _MID_M
        return {k: _W_NEAR[k] + t * (_W_MID[k] - _W_NEAR[k]) for k in _W_NEAR}
    else:
        # Blend from W_MID (r=50) to W_FAR (r=120), then clamp.
        t = min(1.0, (range_m - _MID_M) / (_FAR_M - _MID_M))
        return {k: _W_MID[k] + t * (_W_FAR[k] - _W_MID[k]) for k in _W_MID}


# ---------------------------------------------------------------------------
# Fusion arithmetic helpers
# ---------------------------------------------------------------------------

def _weighted_average(values: list[float], weights: list[float]) -> float:
    """Normalised weighted average of *values* using *weights*.

    Weights are re-normalised internally, so callers may pass raw sensor
    weights without pre-computing their sum.
    """
    total = sum(weights)
    if total == 0.0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total


def _fuse_position(
    positions: list[tuple[float, float]],
    weights: list[float],
) -> FusedPosition:
    """Fuse a list of (x, y) positions via normalised weighted average.

    NaN entries (e.g. a degenerate camera projection) are silently dropped
    so that a bad camera estimate does not corrupt an otherwise good
    radar+lidar position.  If every position is NaN the result is (0, 0)
    with a warning.
    """
    valid_pairs = [
        (pos, w)
        for pos, w in zip(positions, weights)
        if not math.isnan(pos[0]) and not math.isnan(pos[1])
    ]

    if not valid_pairs:
        logger.warning("_fuse_position: all positions are NaN; returning (0, 0)")
        return FusedPosition(x_m=0.0, y_m=0.0)

    xs = [p[0] for p, _ in valid_pairs]
    ys = [p[1] for p, _ in valid_pairs]
    ws = [w    for _, w in valid_pairs]

    return FusedPosition(
        x_m=round(_weighted_average(xs, ws), 3),
        y_m=round(_weighted_average(ys, ws), 3),
    )


def _object_class(
    cam: Optional[CameraDetection],
    rad: Optional[RadarDetection],
    lid: Optional[LidarDetection],
) -> str:
    """Return the best available object class label for classification purposes.

    Priority order
    --------------
    1. Camera ``class_`` — direct semantic classification from the detector.
    2. LiDAR height heuristic — z_m > 1.2 m suggests a pedestrian-height object.
    3. ``'unknown'`` — radar-only tracks carry no class information; these are
       treated as non-VRU by ``_classify()``, which is the conservative choice
       (radar clutter is rarely a pedestrian).
    """
    if cam is not None:
        return cam["class_"]
    if lid is not None:
        return "pedestrian" if lid["z_m"] > 1.2 else "vehicle"
    return "unknown"


def _classify(num_sensors: int, fused_confidence: float, obj_class: str) -> str:
    """Assign a four-tier track status label.

    Parameters
    ----------
    num_sensors:
        Number of distinct sensor modalities that detected this object.
    fused_confidence:
        Weighted-average confidence score across all detecting sensors.
    obj_class:
        Object class string (e.g. ``'vehicle'``, ``'pedestrian'``,
        ``'cyclist'``, ``'unknown'``).  Determines which confirmation
        threshold applies and whether the 'cautionary' tier is eligible.

    Rules (evaluated in priority order)
    ------------------------------------
    confirmed   — ≥ 2 sensors AND confidence ≥ class-appropriate threshold.
                  Uses _VRU_CONFIRMED_THRESHOLD (0.50) for pedestrians and
                  cyclists; _NON_VRU_CONFIRMED_THRESHOLD (0.70) for all other
                  classes.  The lower VRU bar improves recall on the class
                  where a missed detection carries the greatest safety cost.

    tentative   — ≥ 2 sensors AND confidence < class-appropriate threshold.
                  Multiple sensors agree on the object's existence but not
                  confidently enough to act.  Worth maintaining in the track
                  list for temporal promotion.

    cautionary  — Exactly 1 sensor AND obj_class is a VRU AND confidence
                  ≥ _CAUTIONARY_THRESHOLD (0.60).  A single sensor cannot
                  independently confirm a detection, but the safety stakes
                  of a missed VRU justify a precautionary vehicle response
                  (speed reduction, driver alert) below full AEB activation.

    unconfirmed — Exactly 1 sensor AND (non-VRU class OR confidence below
                  _CAUTIONARY_THRESHOLD).  Logged for transparency only.
    """
    is_vru = obj_class in _VRU_CLASSES
    threshold = _VRU_CONFIRMED_THRESHOLD if is_vru else _NON_VRU_CONFIRMED_THRESHOLD

    if num_sensors >= 2:
        return "confirmed" if fused_confidence >= threshold else "tentative"

    # Single-sensor path
    if is_vru and fused_confidence >= _CAUTIONARY_THRESHOLD:
        return "cautionary"
    return "unconfirmed"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fuse_detections(sensor_data: SensorData) -> list[FusedObject]:
    """Run the sensor fusion pipeline over validated multi-sensor detections.

    The pipeline executes the following steps for every unique object_id found
    across all three sensor modalities:

    1. Look up whichever of camera / radar / lidar observed this object_id.
    2. Estimate the object's range from the best available sensor; use that
       range to interpolate a weight set from ``_sensor_weights()``.
    3. Convert each detection to the common ego-vehicle (x, y) frame using the
       per-sensor projection functions above.
    4. Compute a weighted-average fused position, dropping NaN measurements.
    5. Compute a weighted-average fused confidence score.
    6. Determine the object class from the best available sensor.
    7. Classify the track as 'confirmed', 'tentative', 'cautionary', or
       'unconfirmed' using class-dependent thresholds.
    8. Assemble a FusedObject carrying fused metadata plus the raw detections.

    The returned list is sorted by object_id ascending.

    Parameters
    ----------
    sensor_data:
        Validated sensor detections as returned by
        :func:`src.data_loader.load_sensor_data`.

    Returns
    -------
    list[FusedObject]
        One entry per unique object_id across all sensors.

    Implementation note
    -------------------
    Object matching is done here by shared object_id, which is appropriate for
    synthetic data where IDs were pre-assigned.  A real pipeline would instead
    perform data association (e.g. Hungarian algorithm on a Mahalanobis
    distance cost matrix) followed by track management in an Extended Kalman
    Filter (EKF) or Unscented Kalman Filter (UKF) to propagate each track's
    position and velocity estimate across frames.
    """
    # Index each sensor's detections by object_id for O(1) lookup.
    camera_by_id: dict[int, CameraDetection] = {
        d["object_id"]: d for d in sensor_data["camera"]
    }
    radar_by_id: dict[int, RadarDetection] = {
        d["object_id"]: d for d in sensor_data["radar"]
    }
    lidar_by_id: dict[int, LidarDetection] = {
        d["object_id"]: d for d in sensor_data["lidar"]
    }

    all_ids = sorted(camera_by_id.keys() | radar_by_id.keys() | lidar_by_id.keys())

    fused_objects: list[FusedObject] = []

    for oid in all_ids:
        cam = camera_by_id.get(oid)
        rad = radar_by_id.get(oid)
        lid = lidar_by_id.get(oid)

        # Determine the representative range for this object, then derive the
        # interpolated weight set.  Weights shift from lidar-dominant at close
        # range to radar-dominant at long range — see _sensor_weights() and the
        # module docstring for the physical justification.
        rep_range = _representative_range(cam, rad, lid)
        w = _sensor_weights(rep_range)

        # Collect per-sensor positions, confidences, and weights.
        # LiDAR is added first so it contributes most to the fused position
        # when all three sensors are present, consistent with its near-range
        # weight dominance.
        positions:    list[tuple[float, float]] = []
        confidences:  list[float]               = []
        raw_weights:  list[float]               = []
        sensors_seen: list[str]                 = []

        if lid is not None:
            positions.append(_lidar_to_xy(lid))
            confidences.append(lid["confidence"])
            raw_weights.append(w["lidar"])
            sensors_seen.append("lidar")

        if rad is not None:
            positions.append(_radar_to_xy(rad))
            confidences.append(rad["confidence"])
            raw_weights.append(w["radar"])
            sensors_seen.append("radar")

        if cam is not None:
            positions.append(_camera_to_xy(cam))
            confidences.append(cam["confidence"])
            raw_weights.append(w["camera"])
            sensors_seen.append("camera")

        obj_class  = _object_class(cam, rad, lid)
        fused_conf = round(_weighted_average(confidences, raw_weights), 4)
        fused_pos  = _fuse_position(positions, raw_weights)
        status     = _classify(len(sensors_seen), fused_conf, obj_class)

        logger.debug(
            "object_id=%-4d  range=%5.1fm  w=(L=%.2f,R=%.2f,C=%.2f)  "
            "sensors=%-22s  conf=%.3f  status=%-11s  pos=(x=%.1f, y=%.1f)",
            oid, rep_range, w["lidar"], w["radar"], w["camera"],
            "+".join(sensors_seen), fused_conf, status,
            fused_pos["x_m"], fused_pos["y_m"],
        )

        fused_objects.append(
            FusedObject(
                object_id=oid,
                status=status,
                sensors_detected_by=sensors_seen,
                fused_confidence=fused_conf,
                fused_position=fused_pos,
                camera=cam,
                radar=rad,
                lidar=lid,
            )
        )

    confirmed   = sum(1 for o in fused_objects if o["status"] == "confirmed")
    tentative   = sum(1 for o in fused_objects if o["status"] == "tentative")
    cautionary  = sum(1 for o in fused_objects if o["status"] == "cautionary")
    unconfirmed = sum(1 for o in fused_objects if o["status"] == "unconfirmed")

    logger.info(
        "Fusion complete — %d total tracks: %d confirmed, %d tentative, "
        "%d cautionary, %d unconfirmed",
        len(fused_objects), confirmed, tentative, cautionary, unconfirmed,
    )

    return fused_objects
