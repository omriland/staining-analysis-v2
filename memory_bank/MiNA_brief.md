---
title: MiNA — Brief
type: brief
script: MiNA.py
version: 1
---

# MiNA.py — Brief

Purpose
- Manual nucleus outlining + mitochondria skeletonization to quantify footprint, classify individuals vs networks, and estimate branch metrics (networkx-based).

Pipeline
- Load single image → grayscale + normalization → unsharp mask → CLAHE → median → Otsu binary → skeletonize → user outlines nuclei → compute nuclei areas/centroids → build graph from skeleton (8-neighbor) → extract branches → associate components near nuclei → compute per-nucleus metrics → export Excel and figure.

Key constants
- PIXELS_PER_MICRON: 5.7273 → MICRONS_PER_PIXEL = 1/5.7273
- max distance nucleus-to-component: 120 px (association)

Inputs/Outputs
- Input: `.tif/.tiff` (Tk dialog), interactive nucleus outlines
- Output: `<image>_analysis.xlsx` (per nucleus), `<image>_figure.tiff`, console metrics

Notes
- Uses networkx to approximate branch paths; junction detection via convolution-based neighbor counting.
- Designed for single-image, interactive workflow.
