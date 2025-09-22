---
title: Dependencies
type: dependencies
version: 1
---

# Dependencies

Core scientific
- numpy, scipy, pandas

Image processing
- opencv-python, scikit-image, tifffile, imagecodecs

Visualization
- matplotlib

I/O and reports
- openpyxl (Excel writer via pandas), xlsxwriter (used in MiNA scripts)

Graph/network (MiNA variants)
- networkx (MiNA.py)
- skan (MiNA_V2.py)

GUI
- tkinter (standard library), matplotlib TkAgg backend

Notes
- See `requirements.txt` for min versions. Built-ins (tkinter, warnings, traceback, concurrent.futures) are part of Python.
