---
title: Repository Overview
type: overview
version: 1
---

# Repository Overview

This project contains multiple analysis scripts for microscopy staining workflows. Quick inventory:

- analysis_FAST_V2.py
  - Role: Batch blue/red stain counting with proximity filtering and white-balance adjustment; exports per-image CSV, preview PNG, and summary Excel.
  - Input: Folder of `.tif/.tiff` images (RGB/BGR).
  - Output: `analysis_results/` with `<image>.csv`, `<image>_analysis.png`, `stain_count_summary.xlsx`.

- FINAL_ANALYSIS.py
  - Role: GUI for DAPI nuclei counting with manual additions, optional red/green dot analysis, optional nearest-nucleus associations, and multi-sheet Excel report.
  - Input: Folder of `.tif/.tiff` images; interactive parameters.
  - Output: `nuclei_results/` with per-image result images, per-image red/green dot CSVs (if analyzed), and `staining_analysis_report.xlsx`.

- MiNA.py
  - Role: Manual nucleus outlining + mitochondria skeletonization; computes footprint, classifies individuals vs networks, approximates branches with networkx, exports per-nucleus Excel and figures.
  - Input: Single `.tif/.tiff` image; interactive nuclei outlining.
  - Output: `<image>_analysis.xlsx`, `<image>_figure.tiff`.

- MiNA_V2.py
  - Role: Similar to MiNA but uses `skan` for branch metrics; manual nuclei outlining; exports per-nucleus Excel and figures.
  - Input: Single `.tif/.tiff` image; interactive nuclei outlining.
  - Output: `<image>_analysis.xlsx`, `<image>_figure.tiff`.

- nuclei_count_only.py
  - Role: GUI for DAPI nuclei detection only with adjustable parameters and per-image saving; exports summary CSV.
  - Input: Folder of `.tif/.tiff` images.
  - Output: `nuclei_results/` with result images and `nuclei_counts.csv`.

- nuclei_count_with_red_dots.py
  - Role: Extended nuclei GUI with red/green dot detection and total green area metric; interactive per-image params; saves per-image results and green/red data.
  - Input: Folder of `.tif/.tiff` images.
  - Output: `nuclei_results/` with result images and per-image red/green dot CSVs; summary in `FINAL_ANALYSIS.py` when used.

- s129_area.py
  - Role: Threshold-based quantification of red/green area percentages per image; previews thresholds; batch processes a folder.
  - Input: Folder of `.tif/.tiff` images; interactive threshold preview on one sample.
  - Output: CSV `image_analysis_results.csv` with per-image red/green areas and percentages; per-image visualization PNGs.

- S129_BLUE_green_stains.py
  - Role: Counts blue nuclei and green dots per image, saving per-image CSV and preview.
  - Input: Folder of `.tif/.tiff` images.
  - Output: `Results/` with `<image>_analysis.csv` and `<image>_preview.png`.

Notes
- Pixel-to-micron conversion used across scripts: 1 micron = 5.7273 pixels (MICRON_CONVERSION); thus microns per pixel = 1/5.7273.
- GUIs rely on Tkinter and Matplotlib with TkAgg backend.
