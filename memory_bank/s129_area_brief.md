---
title: s129_area — Brief
type: brief
script: s129_area.py
version: 1
---

# s129_area.py — Brief

Purpose
- Quantify percentage area of red and green signals per image using fixed thresholds, with visualization overlays.

Pipeline
- Load image → extract red/green channels → apply thresholds → compute area and percentages → visualize overlays → batch folder processing.

Key parameters
- User-provided thresholds (0–255), previewed interactively on a sample image.

Inputs/Outputs
- Input: Folder of `.tif/.tiff`; user selects input/output folders.
- Output: `image_analysis_results.csv` with per-image Red/Green area and percentage; `<image>_visualization.png` overlays.

Notes
- Uses `skimage.color` to generate channel overlays; thresholds are applied in 0–1 scaled space.
