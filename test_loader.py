"""Quick integration test for src.data_loader."""

import logging
import sys

# Capture WARNING-level log output from the loader so we can surface it in the
# summary rather than just letting it scroll past on stderr.
class _WarningCollector(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


collector = _WarningCollector()
logging.getLogger("src.data_loader").addHandler(collector)
logging.getLogger("src.data_loader").setLevel(logging.WARNING)

from src.data_loader import load_sensor_data  # noqa: E402 (import after logging setup)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
try:
    data = load_sensor_data("data")
except (FileNotFoundError, ValueError) as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)

camera_ids = {d["object_id"] for d in data["camera"]}
radar_ids  = {d["object_id"] for d in data["radar"]}
lidar_ids  = {d["object_id"] for d in data["lidar"]}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 52)
print("  ADAS Data Loader — Summary")
print("=" * 52)

print("\nDetections per sensor")
print(f"  camera : {len(data['camera']):>3} detections   ids: {sorted(camera_ids)}")
print(f"  radar  : {len(data['radar']):>3} detections   ids: {sorted(radar_ids)}")
print(f"  lidar  : {len(data['lidar']):>3} detections   ids: {sorted(lidar_ids)}")

print("\nOverlapping object_ids")
all_three   = camera_ids & radar_ids & lidar_ids
cam_rad     = (camera_ids & radar_ids)  - lidar_ids
cam_lid     = (camera_ids & lidar_ids)  - radar_ids
rad_lid     = (radar_ids  & lidar_ids)  - camera_ids

print(f"  camera + radar + lidar : {sorted(all_three)}")
print(f"  camera + radar only    : {sorted(cam_rad)}")
print(f"  camera + lidar  only   : {sorted(cam_lid)}")
print(f"  radar  + lidar  only   : {sorted(rad_lid)}")

single_cam  = camera_ids - radar_ids - lidar_ids
single_rad  = radar_ids  - camera_ids - lidar_ids
single_lid  = lidar_ids  - camera_ids - radar_ids
print(f"\n  camera-only            : {sorted(single_cam)}")
print(f"  radar-only             : {sorted(single_rad)}")
print(f"  lidar-only             : {sorted(single_lid)}")

print("\nValidation warnings")
if collector.records:
    for msg in collector.records:
        print(f"  WARNING  {msg}")
else:
    print("  none — all records passed validation")

print("=" * 52)
