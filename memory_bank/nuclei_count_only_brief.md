# nuclei_count_only.py — Brief

Purpose
- GUI to count DAPI-stained nuclei with adjustable params and manual additions; saves per-image result images and a summary CSV.

Pipeline
- Load folder → blue channel threshold → morphological closing/dilation → size filter → label contours → draw and number nuclei → optional manual additions.

Key parameters (defaults)
- blue_threshold: 65
- min_size: 153
- max_size: 10000
- dilation_size: 5
- closing_size: 4
- distance_threshold: 15

Inputs/Outputs
- Input: Folder with `.tif/.tiff`
- Output: `nuclei_results/` result images and `nuclei_counts.csv` (Filename, Auto/Manual/Total counts)

Notes
- Per-image parameter overrides are tracked; can apply current params to all images.
