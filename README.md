# ADAS Sensor Fusion Visualizer

## Overview

A lightweight sensor fusion prototype demonstrating a multi-sensor object detection pipeline for autonomous driving. The project takes simulated camera, radar, and lidar detection data, fuses them using range-dependent weighted averaging, and outputs bird's-eye-view and 3D perspective visualizations of the fused scene. Built to demonstrate systems-level thinking about sensor integration and safety-aware classification, not as a production fusion algorithm.

## Motivation

In production ADAS systems, no single sensor provides sufficient reliability for safety-critical decisions. Camera provides rich classification but unreliable depth. Radar gives precise range and velocity but poor angular resolution. Lidar offers accurate 3D geometry but degrades at long range and in adverse weather. Effective fusion must account for the physics-driven reliability characteristics of each sensor modality — and those characteristics change with range, weather, and scenario.

## Features

- **Multi-sensor data loading** with per-modality validation and error handling
- **Range-dependent sensor weighting** reflecting real-world sensor physics: lidar dominates in near range (0–50 m) where point density is high, radar weight increases at mid and far range (50 m+) where its time-of-flight measurement remains consistent, camera contributes classification confidence at all ranges but carries low position weight
- **Class-dependent confidence thresholds** with asymmetric risk handling for Vulnerable Road Users
- **Four-tier detection classification**: confirmed, tentative, cautionary, and unconfirmed
- **Bird's-eye-view and 3D perspective visualization** with range zone indicators and confidence-scaled markers

## Architecture

| Module | Responsibility |
|---|---|
| `src/data_loader.py` | Loads and validates sensor JSON files with per-modality type checking |
| `src/fusion.py` | Matches detections across sensors by object ID, applies range-dependent weighted averaging for confidence and position, classifies detection reliability with class-aware thresholds |
| `src/visualizer.py` | Generates 2D bird's-eye-view and 3D perspective plots with confidence-scaled markers and range zone overlays |
| `main.py` | Orchestrates the full pipeline |

## Key Design Decisions

**Range-dependent weighting:** Fixed sensor weights assume constant reliability across distance, which is physically incorrect. Lidar point density drops with range squared, degrading position estimates at distance. Radar maintains consistent range accuracy via time-of-flight regardless of distance. The weighting function interpolates smoothly between near (0–50 m), mid (50–120 m), and far (120 m+) zones to avoid discontinuities at boundaries. In production systems, these weights would typically be learned from ground-truth data rather than hand-set, but explicit engineering rationale is used here to demonstrate the underlying reasoning.

**Class-dependent confirmation thresholds:** The system applies different confidence thresholds depending on the detected object class. VRU detections (pedestrian, cyclist) use a lower confirmation threshold (0.50) than non-VRU detections (vehicle, 0.70). This reflects the asymmetric risk: the consequence of missing a pedestrian (severe injury or death) is categorically worse than a false positive (unnecessary braking). This asymmetry should drive the threshold design.

**The 'cautionary' classification tier:** A blanket rule requiring multi-sensor corroboration before acting on any detection creates a dangerous gap: in degraded sensor conditions (heavy rain, fog, spray), lidar may scatter and radar may generate clutter, while the camera still detects a pedestrian with reasonable confidence. Discarding that detection because it lacks corroboration ignores the most vulnerable road user on the basis of a rigid architectural rule. The 'cautionary' tier addresses this by flagging single-sensor VRU detections with confidence >= 0.60 for precautionary response (speed reduction, driver alert) without requiring full multi-sensor confirmation. This is not equivalent to full AEB activation — it is a proportional response to uncertain-but-safety-relevant information.

**Single-sensor non-VRU detections remain 'unconfirmed':** For vehicle detections, the risk asymmetry is less extreme and vehicles are larger, more consistent radar/lidar targets. Requiring multi-sensor confirmation for vehicles remains appropriate.

## Confidence Thresholds — A Note on Standards

The thresholds used in this project (0.50 for VRU, 0.70 for non-VRU, 0.60 for cautionary single-sensor VRU) are engineering design decisions justified by risk asymmetry, not values prescribed by any regulatory standard. ISO 26262 defines process rigor through ASIL classification — pedestrian detection failure is typically rated ASIL C/D, demanding the most stringent development and validation processes. Euro NCAP defines system-level AEB performance requirements (detection at specific speeds and scenarios). Neither standard prescribes internal fusion confidence thresholds; these are left to the implementer's safety analysis. Production systems would derive these thresholds through extensive validation against ground-truth test data across diverse operating conditions.

## Real-World Considerations

This is a simplified demonstration. A production sensor fusion system would additionally require: temporal tracking across frames to promote tentative and cautionary detections over time, coordinate frame transformations accounting for sensor mounting positions and vehicle ego-motion, probabilistic association (e.g. Hungarian algorithm or JPDA) rather than ID-based matching, environmental context awareness (wiper state, image-based weather classification) to dynamically adjust sensor availability expectations and confirmation requirements, and handling of additional sensor degradation modes including partial occlusion, sun glare, and sensor misalignment.

## How to Run

```bash
python3 main.py
```

Outputs the fused detection summary to the console and saves visualizations to `output/`.

## Requirements

- Python 3.10+
- matplotlib
- numpy
