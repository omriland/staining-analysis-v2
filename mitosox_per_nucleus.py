#!/usr/bin/env python3
import argparse
import glob
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from tifffile import imread
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, disk, dilation
from skimage.segmentation import watershed
from skimage.measure import label, regionprops_table
from skimage.feature import peak_local_max
from scipy import ndimage as ndi


def load_pair(
    path: str,
    blue_channel: Optional[int],
    red_channel: Optional[int],
    blue_suffix: Optional[str],
    red_suffix: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Returns (blue_img, red_img, basename)
    Supports (a) single multi-channel image or (b) two mono-channel files with suffixes.
    """
    if blue_suffix and red_suffix:
        # separate files mode
        base = path
        blue_path = base + blue_suffix if not base.endswith(blue_suffix) else path
        red_path = base + red_suffix if not base.endswith(red_suffix) else path.replace(blue_suffix, red_suffix)
        if not os.path.exists(blue_path) or not os.path.exists(red_path):
            raise FileNotFoundError(f"Missing pair for base={base}: {blue_path}, {red_path}")
        blue = imread(blue_path)
        red = imread(red_path)
        name = os.path.splitext(os.path.basename(base))[0]
        return blue.astype(np.float32), red.astype(np.float32), name

    # multi-channel mode
    img = imread(path).astype(np.float32)
    name = os.path.splitext(os.path.basename(path))[0]

    if img.ndim == 2:
        raise ValueError(f"Image {path} is single-channel. Provide --blue-suffix/--red-suffix for paired files.")
    if img.ndim == 3:
        # assume HWC or CHW; try to guess
        h, w, c = img.shape[0], img.shape[1], img.shape[2]
        # If it's CHW (channels first), swap
        if c <= 4 and min(img.shape[0], img.shape[1]) <= 6:
            # Probably CHW; swap to HWC
            img = np.moveaxis(img, 0, -1)
            c = img.shape[-1]

        if blue_channel is None or red_channel is None:
            raise ValueError("For multi-channel images, specify --blue-channel and --red-channel (0-based).")
        if blue_channel >= c or red_channel >= c:
            raise ValueError(f"Channel index out of range for {path} with {c} channels.")
        blue = img[..., blue_channel]
        red = img[..., red_channel]
        return blue, red, name

    raise ValueError(f"Unsupported image dimensionality for {path}: {img.shape}")


def estimate_background(img: np.ndarray, method: str = "percentile", p: float = 10.0) -> float:
    if method == "percentile":
        return float(np.percentile(img, p))
    elif method == "mode":
        hist, bin_edges = np.histogram(img.ravel(), bins=256)
        return float(bin_edges[np.argmax(hist)])
    else:
        return 0.0


def segment_nuclei(blue_img: np.ndarray, min_size: int = 50, sigma: float = 1.0) -> np.ndarray:
    # smooth a bit, threshold with Otsu, remove small objects, separate touching with watershed
    sm = gaussian(blue_img, sigma=sigma, preserve_range=True)
    thr = threshold_otsu(sm)
    mask = sm > thr
    mask = remove_small_objects(mask, min_size=min_size)

    # distance and watershed to split touching nuclei
    distance = ndi.distance_transform_edt(mask)
    # find local maxima for markers
    coords = peak_local_max(distance, labels=mask, exclude_border=False, footprint=np.ones((3, 3)))
    markers = np.zeros_like(distance, dtype=int)
    if coords.size > 0:
        for i, (r, c) in enumerate(coords, start=1):
            markers[r, c] = i
    else:
        # fallback: label connected components
        markers = label(mask)

    labels_ws = watershed(-distance, markers=markers, mask=mask)
    return labels_ws.astype(np.int32)


def quantify_per_nucleus(
    labels_img: np.ndarray,
    blue_img: np.ndarray,
    red_img: np.ndarray,
    bg_blue: float,
    bg_red: float,
) -> pd.DataFrame:
    # Background correction (non-negative)
    blue_corr = np.clip(blue_img - bg_blue, a_min=0, a_max=None)
    red_corr = np.clip(red_img - bg_red, a_min=0, a_max=None)

    # Basic regionprops
    props = ("label", "area", "centroid")
    rp = regionprops_table(labels_img, properties=list(props))
    df = pd.DataFrame(rp)

    # Per-label means and sums
    labels = df["label"].to_numpy(dtype=int)

    def per_label_stat(img: np.ndarray, labels_img: np.ndarray, labels: np.ndarray, stat: str) -> np.ndarray:
        out = np.zeros(len(labels), dtype=np.float64)
        for i, lab in enumerate(labels):
            mask = labels_img == lab
            values = img[mask]
            if values.size == 0:
                out[i] = np.nan
            elif stat == "mean":
                out[i] = float(values.mean())
            elif stat == "sum":
                out[i] = float(values.sum())
        return out

    df["hoechst_mean"] = per_label_stat(blue_corr, labels_img, labels, "mean")
    df["hoechst_intden"] = per_label_stat(blue_corr, labels_img, labels, "sum")
    df["mitosox_mean"] = per_label_stat(red_corr, labels_img, labels, "mean")
    df["mitosox_intden"] = per_label_stat(red_corr, labels_img, labels, "sum")

    # Ratios
    df["ratio_mean_red_over_blue"] = df["mitosox_mean"] / (df["hoechst_mean"] + 1e-9)
    df["ratio_intden_red_over_blue"] = df["mitosox_intden"] / (df["hoechst_intden"] + 1e-9)

    return df


def process_folder(
    input_glob: str,
    output_csv: str,
    blue_channel: Optional[int],
    red_channel: Optional[int],
    blue_suffix: Optional[str],
    red_suffix: Optional[str],
    min_nucleus_size: int,
    sigma: float,
    bg_method: str,
    bg_percentile: float,
) -> pd.DataFrame:
    paths = sorted(glob.glob(input_glob))
    if len(paths) == 0:
        raise FileNotFoundError(f"No files matched pattern: {input_glob}")

    all_records: List[pd.DataFrame] = []
    for p in paths:
        try:
            blue, red, name = load_pair(p, blue_channel, red_channel, blue_suffix, red_suffix)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
            continue

        # ensure 2D
        if blue.ndim > 2:
            blue = blue.squeeze()
        if red.ndim > 2:
            red = red.squeeze()

        # Estimate background per image (robust to low-level haze), same for both channels
        bg_b = estimate_background(blue, method=bg_method, p=bg_percentile)
        bg_r = estimate_background(red, method=bg_method, p=bg_percentile)

        labels_img = segment_nuclei(blue, min_size=min_nucleus_size, sigma=sigma)
        df = quantify_per_nucleus(labels_img, blue, red, bg_b, bg_r)
        df.insert(0, "image", name)
        all_records.append(df)

    if not all_records:
        raise RuntimeError("No images were processed successfully. Check your inputs.")

    out = pd.concat(all_records, ignore_index=True)
    out.to_csv(output_csv, index=False)
    return out


def select_input_folder():
    """Open a file dialog to select input folder or files."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Ask user what they want to select
    choice = messagebox.askyesnocancel(
        "Input Selection", 
        "Do you want to select a folder containing images?\n\n"
        "Yes = Select folder\n"
        "No = Select individual files\n"
        "Cancel = Exit"
    )
    
    if choice is None:  # Cancel
        return None, None, None
    
    if choice:  # Yes - select folder
        folder_path = filedialog.askdirectory(title="Select folder containing images")
        if not folder_path:
            return None, None, None
        
        # Ask for file pattern
        pattern = simpledialog.askstring(
            "File Pattern", 
            "Enter file pattern (e.g., *.tif, *.png):", 
            initialvalue="*.tif"
        )
        if not pattern:
            pattern = "*.tif"
        
        return folder_path, pattern, None
    else:  # No - select individual files
        file_paths = filedialog.askopenfilenames(
            title="Select image files",
            filetypes=[
                ("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                ("TIFF files", "*.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if not file_paths:
            return None, None, None
        
        return file_paths, None, None


def select_output_file():
    """Open a file dialog to select output CSV file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    output_path = filedialog.asksaveasfilename(
        title="Save results as",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    return output_path if output_path else "mitosox_per_nucleus.csv"


def get_channel_settings():
    """Get channel settings from user via dialog."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Ask for mode
    mode = messagebox.askyesno(
        "Channel Mode", 
        "Do you have multi-channel images (single file with multiple channels)?\n\n"
        "Yes = Multi-channel mode\n"
        "No = Paired files mode"
    )
    
    if mode:  # Multi-channel mode
        blue_channel = simpledialog.askinteger(
            "Blue Channel", 
            "Enter blue channel index (0-based) for Hoechst:",
            initialvalue=0,
            minvalue=0
        )
        red_channel = simpledialog.askinteger(
            "Red Channel", 
            "Enter red channel index (0-based) for MitoSOX:",
            initialvalue=1,
            minvalue=0
        )
        return blue_channel, red_channel, None, None
    else:  # Paired files mode
        blue_suffix = simpledialog.askstring(
            "Blue Suffix", 
            "Enter suffix for Hoechst files (e.g., _blue.tif):",
            initialvalue="_blue.tif"
        )
        red_suffix = simpledialog.askstring(
            "Red Suffix", 
            "Enter suffix for MitoSOX files (e.g., _red.tif):",
            initialvalue="_red.tif"
        )
        return None, None, blue_suffix, red_suffix


def main():
    ap = argparse.ArgumentParser(description="Per-nucleus MitoSOX quantification normalized to Hoechst.")
    ap.add_argument("--input", help="Input folder path (optional - will prompt if not provided)")
    ap.add_argument("--pattern", default="*.tif", help="Glob pattern inside input folder (default: *.tif)")
    ap.add_argument("--output", help="Output CSV path (optional - will prompt if not provided)")
    # Multi-channel mode options
    ap.add_argument("--blue-channel", type=int, default=None, help="Blue channel index (0-based) for Hoechst")
    ap.add_argument("--red-channel", type=int, default=None, help="Red channel index (0-based) for MitoSOX")
    # Paired-files mode options
    ap.add_argument("--blue-suffix", type=str, default=None, help="Suffix for Hoechst files, e.g., _blue.tif")
    ap.add_argument("--red-suffix", type=str, default=None, help="Suffix for MitoSOX files, e.g., _red.tif")
    # Segmentation/quant params
    ap.add_argument("--min-nucleus-size", type=int, default=50, help="Minimum nucleus area (pixels)")
    ap.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma for smoothing before thresholding")
    # Background options
    ap.add_argument("--bg-method", type=str, choices=["percentile", "mode", "none"], default="percentile",
                    help="Background estimation method")
    ap.add_argument("--bg-percentile", type=float, default=10.0, help="Percentile for background (if method=percentile)")
    ap.add_argument("--gui", action="store_true", help="Use GUI dialogs for input selection")

    args = ap.parse_args()

    # Use GUI if requested or if input not provided
    if args.gui or not args.input:
        print("Opening file selection dialog...")
        
        # Get input selection
        input_result = select_input_folder()
        if input_result[0] is None:
            print("No input selected. Exiting.")
            return
        
        input_path, pattern, file_paths = input_result
        
        # Get output file
        if not args.output:
            output_path = select_output_file()
            if not output_path:
                print("No output file selected. Exiting.")
                return
        else:
            output_path = args.output
        
        # Get channel settings if not provided
        if args.blue_channel is None and args.red_channel is None and args.blue_suffix is None and args.red_suffix is None:
            channel_result = get_channel_settings()
            if channel_result[0] is None and channel_result[2] is None:
                print("No channel settings provided. Exiting.")
                return
            blue_channel, red_channel, blue_suffix, red_suffix = channel_result
        else:
            blue_channel, red_channel, blue_suffix, red_suffix = args.blue_channel, args.red_channel, args.blue_suffix, args.red_suffix
        
        # Set pattern if not provided
        if not pattern:
            pattern = args.pattern
        
        # Process files
        if file_paths:  # Individual files selected
            input_glob = file_paths
        else:  # Folder selected
            input_glob = os.path.join(input_path, pattern)
    else:
        # Use command line arguments
        input_glob = os.path.join(args.input, args.pattern)
        output_path = args.output or "mitosox_per_nucleus.csv"
        blue_channel, red_channel, blue_suffix, red_suffix = args.blue_channel, args.red_channel, args.blue_suffix, args.red_suffix

    print(f"Processing images from: {input_glob}")
    print(f"Output will be saved to: {output_path}")

    df = process_folder(
        input_glob=input_glob,
        output_csv=output_path,
        blue_channel=blue_channel,
        red_channel=red_channel,
        blue_suffix=blue_suffix,
        red_suffix=red_suffix,
        min_nucleus_size=args.min_nucleus_size,
        sigma=args.sigma,
        bg_method=("percentile" if args.bg_method == "percentile" else ("mode" if args.bg_method == "mode" else "none")),
        bg_percentile=args.bg_percentile,
    )

    print(f"Done. Wrote {output_path} with {len(df)} rows.")


if __name__ == "__main__":
    main()
