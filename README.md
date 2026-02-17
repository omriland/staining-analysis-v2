# Staining Analysis v2

Analysis scripts for microscopy staining workflows: nuclei counting, mitochondria network analysis (MiNA), MitoSOX quantification, red/green dot detection, and related pipelines.

## Prerequisites

- **Python 3.8+** (tested with 3.10–3.13)
- **Tkinter** (for GUI scripts): Usually bundled with Python. On Linux, if missing: `sudo apt install python3-tk` (Debian/Ubuntu) or `sudo dnf install python3-tkinter` (Fedora)

## Setup on a New Machine

Each time you use this project on a new computer, create a virtual environment and install dependencies.

### 1. Clone or copy the project

```bash
# If using git
git clone <repo-url>
cd staining-analysis-v2
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run a script

With the virtual environment activated:

```bash
python mitosox_analysis.py
# or
python MiNA_V2.py
# etc.
```

---

## Running Scripts

All scripts are run from the project root with Python. Most open a file/folder picker or GUI when started.

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `mitosox_analysis.py` | MitoSOX (red) + nuclei (blue) per-nucleus quantification | TIFF images | `mitosox_analysis_report.xlsx` |
| `FINAL_ANALYSIS.py` | DAPI nuclei counting with optional red/green dot analysis | Folder of TIFFs | `nuclei_results/`, `staining_analysis_report.xlsx` |
| `nuclei_count_only.py` | DAPI nuclei detection only | Folder of TIFFs | `nuclei_results/`, `nuclei_counts.csv` |
| `nuclei_count_with_red_dots.py` | Nuclei + red/green dot detection | Folder of TIFFs | Per-image results, summary |
| `MiNA.py` | Manual nuclei outlining + mitochondria skeletonization (networkx) | Single TIFF | `*_analysis.xlsx`, `*_figure.tiff` |
| `MiNA_V2.py` | Same as MiNA using skan for branch metrics | Single TIFF | `*_analysis.xlsx`, `*_figure.tiff` |
| `analysis_FAST_V2.py` | Batch blue/red stain counting with proximity filtering | Folder of TIFFs | `analysis_results/`, summary Excel |
| `s129_area.py` | Red/green area percentages per image | Folder of TIFFs | `image_analysis_results.csv`, preview PNGs |
| `S129_BLUE_green_stains.py` | Blue nuclei + green dot counting | Folder of TIFFs | `Results/` with CSV and preview PNG |

---

## Fiji Macros

The `fiji-macros/` folder contains ImageJ/Fiji macros for MitoSOX preprocessing (e.g. `MitoSox_merged_with_results.ijm`). Run these in Fiji/ImageJ if you use that workflow; they are independent of the Python scripts.

---

## Quick Reference

- **Pixel-to-micron conversion**: 1 micron = 5.7273 pixels (used across scripts)
- **Output locations**: Scripts typically write to subfolders like `nuclei_results/`, `analysis_results/`, or `Results/` in the chosen output directory
- **Deactivate venv** when finished: run `deactivate` in the terminal
