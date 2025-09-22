---
title: MitoSOX Per-Nucleus Process
type: process
version: 1
---

# MitoSOX Per-Nucleus Analysis Process

## Overview
MitoSOX per-nucleus analysis is a quantitative microscopy technique that measures oxidative stress in individual cell nuclei by analyzing the MitoSOX Red fluorescent signal normalized to nuclear DNA staining (typically Hoechst).

## Purpose
- Quantify mitochondrial superoxide levels (MitoSOX Red) on a per-nucleus basis
- Normalize the signal to nuclear DNA content (Hoechst blue)
- Provide robust background correction and segmentation
- Support both multi-channel images and paired single-channel files

## Analysis Pipeline

### 1. Image Input
The system supports two modes:
- **Multi-channel mode**: Single file containing multiple channels (e.g., .tif with blue and red channels)
- **Paired files mode**: Separate files for each channel (e.g., `image_blue.tif` and `image_red.tif`)

### 2. Background Estimation
Background correction is crucial for accurate quantification:
- **Percentile method** (default): Uses the 10th percentile of pixel intensities
- **Mode method**: Uses the most frequent intensity value
- **None**: No background correction

### 3. Nuclear Segmentation
Process for identifying individual nuclei from Hoechst (blue) channel:
1. **Gaussian smoothing** (sigma=1.0): Reduces noise
2. **Otsu thresholding**: Automatically determines optimal threshold
3. **Size filtering**: Removes objects smaller than minimum size (default: 50 pixels)
4. **Watershed segmentation**: Separates touching nuclei using:
   - Distance transform to find nuclear centers
   - Local maxima detection for seed points
   - Watershed algorithm to separate boundaries

### 4. Per-Nucleus Quantification
For each segmented nucleus:
1. **Background correction**: Subtract background from both channels
2. **Measurements extracted**:
   - Area (pixels)
   - Centroid coordinates
   - Hoechst mean intensity
   - Hoechst integrated density (sum of all pixel values)
   - MitoSOX mean intensity
   - MitoSOX integrated density
3. **Calculated ratios**:
   - Mean intensity ratio: MitoSOX/Hoechst
   - Integrated density ratio: MitoSOX/Hoechst

### 5. Output Format
CSV file containing per-nucleus data:
- `image`: Source image filename
- `label`: Unique nucleus identifier
- `area`: Nuclear area in pixels
- `centroid-0`, `centroid-1`: X,Y coordinates
- `hoechst_mean`: Mean Hoechst intensity
- `hoechst_intden`: Total Hoechst intensity
- `mitosox_mean`: Mean MitoSOX intensity
- `mitosox_intden`: Total MitoSOX intensity
- `ratio_mean_red_over_blue`: Normalized mean intensity
- `ratio_intden_red_over_blue`: Normalized integrated density

## Key Parameters

### Segmentation Parameters
- `min_nucleus_size`: Minimum nuclear area (default: 50 pixels)
- `sigma`: Gaussian smoothing parameter (default: 1.0)

### Background Correction
- `bg_method`: Method for background estimation ('percentile', 'mode', 'none')
- `bg_percentile`: Percentile for background if using percentile method (default: 10.0)

### Channel Configuration
For multi-channel images:
- `blue_channel`: Index of Hoechst channel (0-based)
- `red_channel`: Index of MitoSOX channel (0-based)

For paired files:
- `blue_suffix`: File suffix for Hoechst images (e.g., '_blue.tif')
- `red_suffix`: File suffix for MitoSOX images (e.g., '_red.tif')

## Usage Modes

### GUI Mode
- Interactive file/folder selection
- Channel configuration dialogs
- User-friendly for single analyses

### Command Line Mode
- Batch processing capability
- Scriptable for automation
- Supports glob patterns for file selection

## Best Practices
1. Ensure consistent imaging conditions across samples
2. Use appropriate exposure times to avoid saturation
3. Include proper controls (unstained, single-stained)
4. Verify segmentation quality on representative images
5. Consider biological variability when setting parameters


