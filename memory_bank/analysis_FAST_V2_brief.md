---
title: analysis_FAST_V2 — Brief
type: brief
script: analysis_FAST_V2.py
version: 1
---

# analysis_FAST_V2.py — Brief

Purpose
- Batch count blue nuclei and red stains with proximity to blue, with white-balance adjustment and size filtering. Exports per-image CSV/PNG and a summary Excel.

Pipeline
- Read RGB image → CLAHE on L channel (LAB) → detect blue/red via channel threshold → remove small objects → proximity filter red to be within distance of blue → count nuclei and red props → output.

Key parameters (defaults via prompts)
- min_blue_size: 500
- proximity_distance: 150
- blue_threshold: 50
- red_threshold: 120
- min_red_size: 2
- MICRON_CONVERSION: 5.7273 px/μm (used for red object size reporting in CSV)

Inputs/Outputs
- Input: Folder with `.tif/.tiff`
- Output: `<folder>/analysis_results/` with:
  - `<image>.csv` (counts, red_i size μm²)
  - `<image>_analysis.png` preview
  - `stain_count_summary.xlsx` across images

Notes
- Multiprocessing used for per-file processing.
- Excel generation requires pandas+openpyxl; falls back to CSV if missing.
