---
title: MiNA_V2 — Brief
type: brief
script: MiNA_V2.py
version: 1
---

# MiNA_V2.py — Brief

Purpose
- Manual nucleus outlining + mitochondria skeletonization using `skan` to compute branch path lengths and related metrics; per-nucleus Excel and figure outputs.

Pipeline
- Load single image → grayscale + normalization → unsharp mask → CLAHE → median → Otsu binary → skeletonize → user outlines nuclei → nuclei areas/centroids → `skan.csr.Skeleton` for branch lengths → classify individuals vs networks via junction pixels → associate components near nuclei → compute per-nucleus metrics → export.

Key constants
- PIXELS_PER_MICRON: 5.7273 → MICRONS_PER_PIXEL = 1/5.7273
- Association max distance nucleus-to-component: ~120 px

Inputs/Outputs
- Input: `.tif/.tiff` (Tk dialog), interactive nucleus outlines
- Output: `<image>_analysis.xlsx`, `<image>_figure.tiff`, console metrics

Notes
- Requires `skan`; if unavailable, see `MiNA.py` which uses networkx instead.
