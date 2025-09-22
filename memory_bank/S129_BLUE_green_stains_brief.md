# S129_BLUE_green_stains.py — Brief

Purpose
- Count blue nuclei and green dots per image; export per-image CSV and preview visuals.

Pipeline
- Detect blue via blue channel threshold → size filter for nuclei → count.
- Detect green via green channel threshold → label components → count and compute per-dot size (μm²) → preview overlays.

Key parameters (defaults)
- min_blue_size: 1000
- blue_threshold: 50
- green_threshold: 150
- MICRON_CONVERSION: 5.7273 px/μm (used to compute dot sizes)

Inputs/Outputs
- Input: Folder of `.tif/.tiff`
- Output: `Results/` with `<image>_analysis.csv` (blue/green counts + green sizes) and `<image>_preview.png`.

Notes
- Simple threshold-based approach; no proximity mapping to nuclei.
