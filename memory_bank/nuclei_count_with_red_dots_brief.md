# nuclei_count_with_red_dots.py — Brief

Purpose
- GUI to count nuclei and detect red (and optionally green) dots; supports nearest-nucleus associations and total green area; exports per-image outputs and aggregates.

Pipeline
- Nuclei: same as nuclei_count_only (threshold + morphology + size) + manual additions.
- Red dots: blur + threshold + size filter; store centroid, area (pixels, μm²), intensity.
- Green: always compute total green area (thresholded), optionally detect individual green dots with size filtering.
- Optional: nearest nucleus mapping for dots and per-nucleus summaries.

Key parameters (notable defaults)
- blue_threshold: 65; min/max_size, dilation/closing, distance_threshold
- red_threshold: 100; red_min_size: 5; red_max_size: 200
- If green analysis on: green_threshold: 100; green_min_size: 5; green_max_size: 200
- MICRON_CONVERSION: 5.7273 px/μm

Inputs/Outputs
- Input: Folder with `.tif/.tiff`
- Output: `nuclei_results/` result images; red_dots_<image>.csv; green_dots_<image>.csv (if enabled). Aggregates written by `FINAL_ANALYSIS.py`.

Notes
- UI allows per-image param overrides; can apply to all.
- Maintains manual nuclei markers per image.
