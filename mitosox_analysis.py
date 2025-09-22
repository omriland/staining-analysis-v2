#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MitoSOX Analysis Application
# This script analyzes MitoSOX (red) and nuclei (blue) channels in TIFF images
# For each nucleus, it measures MitoSOX intensity and area statistics
# Based on the structure of FINAL_ANALYSIS.py but adapted for MitoSOX quantification

# Required imports
import os
import sys
import traceback
import numpy as np
import pandas as pd
import cv2
from skimage import io, morphology, measure, segmentation, filters
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import matplotlib
import warnings

# Ignore openpyxl warning about default styles
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl.styles')

matplotlib.use('TkAgg')


class MitoSOXAnalyzer:
    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions by showing a message box with the error details"""
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print("ERROR:", error_msg)
        messagebox.showerror("Application Error", 
                            f"An error occurred:\n\n{exc_value}\n\nCheck console for full details.")
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        
    def start_analysis(self):
        """Start the main analysis after initial options are selected"""
        try:
            print("Starting MitoSOX analysis...")
            
            # Destroy initial dialog
            self.initial_frame.destroy()
            
            # Reset window size for main application
            self.root.geometry("1400x900")
            
            # Create main GUI
            self.create_main_gui()
            print("Main GUI created successfully")

        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Error in start_analysis: {e}\n{error_details}")
            messagebox.showerror("Error Starting Analysis", 
                                f"An error occurred while starting the analysis: {e}\n\nSee console for details.")
            self.recreate_initial_dialog()
        
    def recreate_initial_dialog(self):
        """Recreate the initial dialog if there was an error in start_analysis"""
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Reset window size
        self.root.geometry("400x200")
        
        # Recreate the initial dialog
        self.initial_frame = ttk.Frame(self.root, padding=20)
        self.initial_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(self.initial_frame, text="MitoSOX Analysis Tool", font=("Arial", 14, "bold")).pack(pady=(0, 15))
        
        ttk.Label(self.initial_frame, text="This tool analyzes MitoSOX (red) and nuclei (blue) channels", 
                 font=("Arial", 10)).pack(pady=5)
        
        ttk.Label(self.initial_frame, text="For each nucleus, it measures MitoSOX intensity and area", 
                 font=("Arial", 10)).pack(pady=5)
        
        ttk.Button(self.initial_frame, text="Start Analysis", command=self.start_analysis).pack(pady=15)
        
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MitoSOX Analysis Tool")
        
        # Set up exception handling
        sys.excepthook = self.handle_exception
        
        # Start with a small window size for the initial dialog
        self.root.geometry("400x200")
        self.root.update_idletasks()
        
        # Center the initial dialog
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 400) // 2
        y = (screen_height - 200) // 2
        self.root.geometry(f"400x200+{x}+{y}")
        
        # Initialize data storage
        self.nuclei_data = {}  # Store nucleus measurements for each image
        self.current_image = None
        self.current_filename = None
        self.input_dir = None
        self.output_dir = None
        self.tiff_files = []
        self.current_file_index = 0
        
        # Initialize results storage
        self.total_results = []
        self.manual_nuclei = []
        self.stored_manual_nuclei = {}
        
        # Initialize UI elements
        self.preview_fig = None
        self.preview_axs = None
        self.canvas = None
        self.param_frame = None
        self.sliders = {}
        self.param_modified = False
        
        # Default parameters for MitoSOX analysis
        self.default_params = {
            'nuclei_threshold': 65,  # Threshold for blue nuclei channel
            'nuclei_min_size': 153,  # Minimum nucleus size in pixels
            'nuclei_max_size': 10000,  # Maximum nucleus size in pixels
            'nuclei_dilation_size': 5,  # Size of dilation kernel
            'nuclei_closing_size': 4,  # Size of closing kernel
            'nuclei_distance_threshold': 15,  # Distance threshold for connecting fragments
            'mitosox_threshold': 100,  # Threshold for red MitoSOX channel
            'mitosox_min_size': 5,  # Minimum MitoSOX signal size
            'mitosox_max_size': 200  # Maximum MitoSOX signal size
        }
        
        # Micron conversion factor (adjust based on your microscope settings)
        self.MICRON_CONVERSION = 5.7273  # 1 micron = 5.7273 pixels
        
        # Current parameters (will be image-specific)
        self.params = self.default_params.copy()
        
        # Store image-specific parameters
        self.image_params = {}
        
        # Create initial dialog
        self.initial_frame = ttk.Frame(self.root, padding=20)
        self.initial_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(self.initial_frame, text="MitoSOX Analysis Tool", font=("Arial", 14, "bold")).pack(pady=(0, 15))
        
        ttk.Label(self.initial_frame, text="This tool analyzes MitoSOX (red) and nuclei (blue) channels", 
                 font=("Arial", 10)).pack(pady=5)
        
        ttk.Label(self.initial_frame, text="For each nucleus, it measures MitoSOX intensity and area", 
                 font=("Arial", 10)).pack(pady=5)
        
        ttk.Button(self.initial_frame, text="Start Analysis", command=self.start_analysis).pack(pady=15)
        
        # Run the main loop
        self.root.mainloop()
        
    def create_main_gui(self):
        """Create the main GUI with all necessary frames"""
        # Set window size
        self.root.geometry("1400x900")

        # Create main frames
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create header with file selection
        header_frame = ttk.Frame(self.main_frame, padding=5)
        header_frame.pack(fill=tk.X, pady=5)

        ttk.Label(header_frame, text="MitoSOX Analysis Tool", font=("Arial", 16, "bold")).pack(side=tk.LEFT, padx=5)

        select_btn = ttk.Button(header_frame, text="Select Folder with TIFF Images",
                                command=self.select_and_load_folder)
        select_btn.pack(side=tk.RIGHT, padx=5)

        # Create main content frame
        content_frame = ttk.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Left frame for parameters - set fixed width
        self.param_frame = ttk.LabelFrame(content_frame, text="Parameters", padding=10, width=500)
        self.param_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.param_frame.pack_propagate(False)  # Prevent frame from shrinking

        # Initialize param_status for tracking parameter changes
        self.param_status = ttk.Label(self.param_frame, text="Using default parameters")
        self.param_status.pack(anchor="w", pady=5)

        # Right frame for previews and navigation
        preview_frame = ttk.LabelFrame(content_frame, text="Preview & Navigation", padding=10)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create navigation header
        nav_frame = ttk.Frame(preview_frame)
        nav_frame.pack(fill=tk.X, pady=5)

        # Current image indicator
        self.status_label = ttk.Label(nav_frame, text="No images loaded")
        self.status_label.pack(side=tk.LEFT, pady=5)

        # Nuclei count indicator
        self.count_label = ttk.Label(nav_frame, text="Nuclei count: 0", font=("Arial", 12, "bold"))
        self.count_label.pack(side=tk.RIGHT, pady=5)
        
        # Navigation buttons
        buttons_frame = ttk.Frame(preview_frame)
        buttons_frame.pack(fill=tk.X, pady=10)

        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        buttons_frame.columnconfigure(2, weight=1)
        buttons_frame.columnconfigure(3, weight=1)

        self.prev_btn = ttk.Button(buttons_frame, text="← Previous", command=self.move_to_prev_image, state=tk.DISABLED,
                                  width=15)
        self.prev_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.save_btn = ttk.Button(buttons_frame, text="Save Current", command=self.save_current_image,
                                  state=tk.DISABLED, width=15)
        self.save_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.next_btn = ttk.Button(buttons_frame, text="Save & Next →", command=self.move_to_next_image,
                                  state=tk.DISABLED, width=15)
        self.next_btn.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.process_all_btn = ttk.Button(buttons_frame, text="Process All Remaining", command=self.process_remaining,
                                         state=tk.DISABLED, width=15)
        self.process_all_btn.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        # Canvas for matplotlib
        self.canvas_frame = ttk.Frame(preview_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Add a footer status frame
        status_frame = ttk.Frame(preview_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        self.ui_status = ttk.Label(status_frame, text="Please select a folder with TIFF images", 
                                  font=("Arial", 12), foreground="blue")
        self.ui_status.pack(side=tk.LEFT, pady=5, padx=10)

    def select_and_load_folder(self):
        """Select a folder and load the first image"""
        input_dir = filedialog.askdirectory(title="Select folder with TIFF images")
        if not input_dir:
            print("No folder selected.")
            return

        # Create output directory
        self.output_dir = os.path.join(input_dir, "mitosox_results")
        os.makedirs(self.output_dir, exist_ok=True)

        # Get list of TIFF files
        tiff_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))]
        if not tiff_files:
            messagebox.showerror("Error", f"No TIFF files found in {input_dir}.")
            return

        print(f"Found {len(tiff_files)} TIFF files in {input_dir}")

        self.input_dir = input_dir
        self.tiff_files = tiff_files
        self.current_file_index = 0

        # Load first image
        success = self.load_current_image()

        if success:
            print(f"Loaded first image: {self.current_filename}")
            # Enable buttons
            self.save_btn.config(state=tk.NORMAL)
            self.next_btn.config(state=tk.NORMAL)
            self.process_all_btn.config(state=tk.NORMAL)
            print("Buttons have been enabled")
            
            self.ui_status.config(text="Image loaded - Ready to process!", foreground="green")
            
            # Create parameter sliders
            self.create_parameter_sliders()
        else:
            messagebox.showerror("Error", "Failed to load the first image.")
            print("Failed to load the first image.")
            self.ui_status.config(text="Error loading image. Please try again.", foreground="red")

    def load_current_image(self):
        """Load the current image based on index"""
        print("Starting load_current_image...")
        if not self.tiff_files or self.current_file_index >= len(self.tiff_files):
            print("No files to load or index out of range")
            return False

        filename = self.tiff_files[self.current_file_index]
        image_path = os.path.join(self.input_dir, filename)
        print(f"Attempting to load image: {image_path}")

        # Store current manual nuclei before changing images
        if self.current_filename and self.manual_nuclei:
            self.stored_manual_nuclei[self.current_filename] = self.manual_nuclei.copy()

        # Load new image
        self.current_filename = filename
        try:
            # Read the image using scikit-image (loads in RGB format)
            image = io.imread(image_path)
            print(f"Image loaded successfully. Shape: {image.shape}")

            # Handle different image formats
            if len(image.shape) == 2:  # Grayscale
                print("Detected grayscale image")
                blue_channel = image
                red_channel = None
            elif len(image.shape) == 3:  # RGB or RGBA
                print(f"Detected color image with {image.shape[2]} channels")
                if image.shape[2] >= 3:  # At least 3 channels (RGB)
                    blue_channel = image[:, :, 2]  # blue channel is index 2 in RGB
                    red_channel = image[:, :, 0]   # red channel is index 0 in RGB
                    print("Extracted blue channel (index 2) and red channel (index 0)")
                else:
                    raise ValueError(f"Unexpected number of channels: {image.shape[2]}")
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")

            self.current_image = (image, blue_channel, red_channel)
            print("Current image set successfully")

            # Debug information about loaded image
            print(f"Current image dimensions: {image.shape}")
            print(f"Blue channel dimensions: {blue_channel.shape}")
            if red_channel is not None:
                print(f"Red channel dimensions: {red_channel.shape}")
                print(f"Red channel min/max: {red_channel.min()}/{red_channel.max()}")
            print(f"Blue channel min/max: {blue_channel.min()}/{blue_channel.max()}")

        except Exception as e:
            print(f"Error loading image: {str(e)}")
            messagebox.showerror("Error", f"Failed to load image {filename}: {e}")
            return False

        # Retrieve any stored manual nuclei for this image
        if filename in self.stored_manual_nuclei:
            self.manual_nuclei = self.stored_manual_nuclei[filename].copy()
        else:
            self.manual_nuclei = []
            

        # Load image-specific parameters if they exist
        if filename in self.image_params:
            self.params = self.image_params[filename].copy()
            self.param_modified = True
            self.param_status.config(text="* This image has custom parameters *")
        else:
            self.params = self.default_params.copy()
            self.param_modified = False
            self.param_status.config(text="Using default parameters")

        # Update sliders if they exist
        if hasattr(self, 'sliders') and self.sliders:
            for param, (var, _, _) in self.sliders.items():
                var.set(str(self.params[param]))

        # Update navigation status
        current_idx = self.current_file_index + 1
        self.status_label.config(text=f"Image {current_idx} of {len(self.tiff_files)}: {filename}")

        # Update preview
        print("Initializing preview...")
        self.initialize_preview()
        print("Preview initialized")

        # Update navigation buttons
        self.prev_btn.config(state=tk.NORMAL if self.current_file_index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_file_index < len(self.tiff_files) - 1 else tk.DISABLED)

        return True

    def create_parameter_sliders(self):
        """Create parameter input fields for MitoSOX analysis"""
        # Clear existing widgets in the parameter frame, except param_status
        for widget in self.param_frame.winfo_children():
            if widget != self.param_status:
                widget.destroy()

        # Parameter descriptions for MitoSOX analysis
        param_descriptions = {
            'nuclei_threshold': "Pixel brightness threshold for detecting blue nuclei.\nHigher values detect fewer, brighter nuclei.",
            'nuclei_min_size': "Minimum size in pixels for a valid nucleus.\nIncrease to filter out small artifacts.",
            'nuclei_max_size': "Maximum size in pixels for a valid nucleus.\nIncrease to include larger cell clusters.",
            'nuclei_dilation_size': "Size of dilation kernel to connect fragments.\nLarger values connect more fragments.",
            'nuclei_closing_size': "Size of closing kernel for morphological operations.\nHelps fill holes in detected regions.",
            'nuclei_distance_threshold': "Distance threshold for connecting nearby fragments.\nLarger values merge more regions.",
            'mitosox_threshold': "Pixel brightness threshold for detecting MitoSOX signals.\nHigher values detect fewer, brighter signals.",
            'mitosox_min_size': "Minimum size in pixels for a valid MitoSOX signal.\nIncrease to filter out small artifacts.",
            'mitosox_max_size': "Maximum size in pixels for a valid MitoSOX signal.\nIncrease to include larger signals."
        }

        # Add explanatory text
        info_frame = ttk.Frame(self.param_frame, borderwidth=2, relief="groove", padding=10)
        info_frame.pack(fill=tk.X, pady=10)

        ttk.Label(info_frame, text="MitoSOX Analysis Parameters", font=("Arial", 12, "bold"), foreground="blue").pack(anchor="w")

        ttk.Label(
            info_frame,
            text="• Adjust parameters for nucleus detection and MitoSOX quantification\n"
                 "• Parameters affect only the current image unless 'Apply to All' is used\n"
                 "• Use up/down arrow keys when focused to adjust values",
            font=("Arial", 10),
            justify="left"
        ).pack(anchor="w", pady=5)

        # Re-pack the param_status for current image
        self.param_status.pack(anchor="w", pady=5)

        # Create frame for input fields
        inputs_frame = ttk.Frame(self.param_frame)
        inputs_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Add column headers
        ttk.Label(inputs_frame, text="Parameter", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        ttk.Label(inputs_frame, text="Value", font=("Arial", 10, "bold")).grid(row=0, column=2, sticky=tk.W, pady=(0, 10))
        ttk.Label(inputs_frame, text="Default", font=("Arial", 10, "bold")).grid(row=0, column=3, sticky=tk.W, pady=(0, 10))

        self.sliders = {}
        row = 1

        # Create input fields for each parameter
        for param, value in self.params.items():
            # Parameter label with info icon
            label_frame = ttk.Frame(inputs_frame)
            label_frame.grid(row=row, column=0, sticky=tk.W, pady=5)
            
            label = ttk.Label(label_frame, text=param.replace('_', ' ').title())
            label.pack(side=tk.LEFT)
            
            info_icon = ttk.Label(label_frame, text=" ⓘ", foreground="blue", cursor="hand2")
            info_icon.pack(side=tk.LEFT, padx=(2, 0))
            
            # Create tooltip for info icon
            tooltip = None
            
            def show_tooltip(event, description=param_descriptions[param], widget=info_icon):
                x, y, _, _ = widget.bbox("insert")
                x += widget.winfo_rootx() + 15
                y += widget.winfo_rooty() + 10
                
                nonlocal tooltip
                tooltip = tk.Toplevel(widget)
                tooltip.wm_overrideredirect(True)
                tooltip.wm_geometry(f"+{x}+{y}")
                
                label = ttk.Label(tooltip, text=description, justify=tk.LEFT,
                                background="#ffffaa", relief="solid", borderwidth=1,
                                padding=(5, 3))
                label.pack()
            
            def hide_tooltip(event):
                nonlocal tooltip
                if tooltip:
                    tooltip.destroy()
                    tooltip = None
            
            info_icon.bind("<Enter>", show_tooltip)
            info_icon.bind("<Leave>", hide_tooltip)

            # Create variable for the input
            var = tk.StringVar(value=str(value))
            
            # Create entry widget
            entry = ttk.Entry(inputs_frame, textvariable=var, width=10)
            entry.grid(row=row, column=2, padx=5, pady=5)
            
            # Show default value
            default_label = ttk.Label(inputs_frame, text=str(self.default_params[param]))
            default_label.grid(row=row, column=3, padx=5, pady=5)
            
            # Add validation
            def validate_input(P):
                if P == "": return True
                try:
                    float(P)
                    return True
                except ValueError:
                    return False
            
            vcmd = (self.root.register(validate_input), '%P')
            entry.config(validate='key', validatecommand=vcmd)
            
            # Add up/down arrow key bindings
            def on_up_arrow(event):
                try:
                    current = float(var.get() or 0)
                    new_value = int(current + 1)
                    var.set(str(new_value))
                    self.update_param(param, new_value)
                except ValueError:
                    pass
                return "break"
            
            def on_down_arrow(event):
                try:
                    current = float(var.get() or 0)
                    new_value = int(current - 1)
                    if new_value >= 0:
                        var.set(str(new_value))
                        self.update_param(param, new_value)
                except ValueError:
                    pass
                return "break"
            
            entry.bind('<Up>', on_up_arrow)
            entry.bind('<Down>', on_down_arrow)
            
            # Add trace for value changes with immediate update
            def on_value_change(*args, p=param, v=var):
                try:
                    value = v.get().strip()
                    if value:
                        new_value = int(float(value))
                        self.update_param(p, new_value)
                except ValueError:
                    pass
            
            var.trace_add("write", on_value_change)
            
            # Add validation and immediate update on Enter key
            def on_enter(event):
                try:
                    value = var.get().strip()
                    if value:
                        new_value = int(float(value))
                        self.update_param(param, new_value)
                except ValueError:
                    var.set(str(self.params[param]))
                return "break"
            
            entry.bind('<Return>', on_enter)
            
            # Add focus out handler to validate and update
            def on_focus_out(event):
                try:
                    value = var.get().strip()
                    if value:
                        new_value = int(float(value))
                        self.update_param(param, new_value)
                except ValueError:
                    var.set(str(self.params[param]))
            
            entry.bind('<FocusOut>', on_focus_out)
            
            # Add reset button
            reset_btn = ttk.Button(
                inputs_frame,
                text="Reset",
                width=6,
                command=lambda p=param: self.reset_param_to_default(p)
            )
            reset_btn.grid(row=row, column=4, padx=5, pady=5)
            
            self.sliders[param] = (var, entry, reset_btn)
            row += 1

        # Parameter action buttons
        actions_frame = ttk.Frame(self.param_frame)
        actions_frame.pack(fill=tk.X, pady=10)

        reset_all_btn = ttk.Button(
            actions_frame,
            text="Reset All to Default",
            command=self.reset_all_params
        )
        reset_all_btn.pack(side=tk.LEFT, padx=5)

        apply_all_btn = ttk.Button(
            actions_frame,
            text="Apply to All Images",
            command=self.apply_params_to_all
        )
        apply_all_btn.pack(side=tk.LEFT, padx=5)

        clear_manual_btn = ttk.Button(self.param_frame, text="Clear Manual Markers", command=self.clear_manual_nuclei)
        clear_manual_btn.pack(fill=tk.X, pady=10)

    def update_param(self, param, value):
        """Update parameter value and refresh preview"""
        prev_value = self.params[param]
        self.params[param] = value

        if value != self.default_params[param]:
            self.param_modified = True
            self.image_params[self.current_filename] = self.params.copy()
            if hasattr(self, 'param_status'):
                self.param_status.config(text="* This image has custom parameters *")
        elif prev_value != value:
            all_default = all(self.params[p] == self.default_params[p] for p in self.params)
            if all_default:
                if self.current_filename in self.image_params:
                    del self.image_params[self.current_filename]
                self.param_modified = False
                if hasattr(self, 'param_status'):
                    self.param_status.config(text="Using default parameters")

        self.update_preview()

    def reset_param_to_default(self, param):
        """Reset a specific parameter to its default value"""
        default_value = self.default_params[param]
        var, _, _ = self.sliders[param]
        var.set(str(default_value))

    def reset_all_params(self):
        """Reset all parameters to default values"""
        for param, (var, _, _) in self.sliders.items():
            var.set(str(self.default_params[param]))
        
        if self.current_filename in self.image_params:
            del self.image_params[self.current_filename]
        
        self.param_modified = False
        self.param_status.config(text="Using default parameters")

    def apply_params_to_all(self):
        """Apply current parameters to all images"""
        confirm = messagebox.askyesno(
            "Apply to All Images",
            "Are you sure you want to apply the current parameters to ALL images?\n"
            "This will override any custom settings for individual images."
        )

        if confirm:
            self.default_params = self.params.copy()
            self.image_params = {}
            self.param_modified = False
            self.param_status.config(text="Using default parameters (applied to all)")

            messagebox.showinfo(
                "Parameters Applied",
                "Current parameters have been set as the new default for all images."
            )

    def clear_manual_nuclei(self):
        """Clear all manually added nuclei markers"""
        self.manual_nuclei = []
        self.update_preview()

    def process_image(self, image, blue_channel, red_channel):
        """Process image to detect nuclei and analyze MitoSOX signals per nucleus"""
        # Process nuclei detection (similar to FINAL_ANALYSIS.py)
        blurred_blue = cv2.GaussianBlur(blue_channel, (5, 5), 0)
        _, binary = cv2.threshold(blurred_blue, self.params['nuclei_threshold'], 255, cv2.THRESH_BINARY)

        # Apply morphological operations
        if self.params['nuclei_closing_size'] > 0:
            closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (self.params['nuclei_closing_size'], self.params['nuclei_closing_size']))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, closing_kernel)

        if self.params['nuclei_dilation_size'] > 0:
            dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (self.params['nuclei_dilation_size'], self.params['nuclei_dilation_size']))
            binary = cv2.dilate(binary, dilation_kernel, iterations=1)
            binary = cv2.erode(binary, dilation_kernel, iterations=1)

        # Apply distance threshold to merge nearby objects
        distance_threshold = self.params['nuclei_distance_threshold']
        if distance_threshold > 1:
            dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            _, dist_thresh = cv2.threshold(dist_transform, distance_threshold, 255, cv2.THRESH_BINARY)
            dist_thresh = np.uint8(dist_thresh)
            markers = measure.label(dist_thresh)
            binary = segmentation.watershed(-dist_transform, markers, mask=binary)
            binary = np.uint8(binary > 0) * 255

        # Label connected components for nuclei
        labels = measure.label(binary)
        regions = measure.regionprops(labels)

        # Filter by size and create a mask for valid nuclei
        valid_labels = [region.label for region in regions
                        if self.params['nuclei_min_size'] <= region.area <= self.params['nuclei_max_size']]

        nuclei_mask = np.zeros_like(binary)
        for label in valid_labels:
            nuclei_mask[labels == label] = 255

        # Create a result image with outlines
        if len(image.shape) == 2:
            result_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result_img = image.copy()

        # Find contours of nuclei
        contours, _ = cv2.findContours(nuclei_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Store nucleus data for analysis
        nuclei_data = []
        
        # Draw contours and number them
        nuclei_color = (255, 255, 0)  # Cyan color for blue nuclei in BGR format
        line_thickness = 2
        
        for i, contour in enumerate(contours):
            cv2.drawContours(result_img, [contour], -1, nuclei_color, line_thickness)
            
            # Calculate nucleus properties
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Calculate nucleus area
                nucleus_area_pixels = cv2.contourArea(contour)
                nucleus_area_microns = nucleus_area_pixels / (self.MICRON_CONVERSION * self.MICRON_CONVERSION)
                
                # Calculate nucleus intensity statistics from blue channel
                mask = np.zeros(blue_channel.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                nucleus_pixels = blue_channel[mask > 0]
                
                nucleus_mean = np.mean(nucleus_pixels) if len(nucleus_pixels) > 0 else 0
                nucleus_min = np.min(nucleus_pixels) if len(nucleus_pixels) > 0 else 0
                nucleus_max = np.max(nucleus_pixels) if len(nucleus_pixels) > 0 else 0
                
                # Analyze MitoSOX signal within this nucleus
                mitosox_mean = 0
                mitosox_min = 0
                mitosox_max = 0
                mitosox_area_pixels = 0
                mitosox_area_microns = 0
                
                if red_channel is not None:
                    # Extract MitoSOX signal within the nucleus region
                    mitosox_pixels = red_channel[mask > 0]
                    
                    if len(mitosox_pixels) > 0:
                        mitosox_mean = np.mean(mitosox_pixels)
                        mitosox_min = np.min(mitosox_pixels)
                        mitosox_max = np.max(mitosox_pixels)
                        
                        # Detect MitoSOX signals above threshold within nucleus
                        mitosox_mask = (red_channel > self.params['mitosox_threshold']) & (mask > 0)
                        
                        if np.any(mitosox_mask):
                            mitosox_labels = measure.label(mitosox_mask.astype(np.uint8))
                            mitosox_regions = measure.regionprops(mitosox_labels)
                            
                            # Filter MitoSOX signals by size
                            valid_mitosox_labels = [region.label for region in mitosox_regions
                                                   if self.params['mitosox_min_size'] <= region.area <= self.params['mitosox_max_size']]
                            
                            mitosox_area_pixels = sum(region.area for region in mitosox_regions 
                                                     if region.label in valid_mitosox_labels)
                            mitosox_area_microns = mitosox_area_pixels / (self.MICRON_CONVERSION * self.MICRON_CONVERSION)
                            
                            # Draw MitoSOX signals
                            for region in mitosox_regions:
                                if region.label in valid_mitosox_labels:
                                    cy_mito, cx_mito = region.centroid
                                    cv2.circle(result_img, (int(cx_mito), int(cy_mito)), 3, (0, 0, 255), -1)  # Red for MitoSOX
                
                # Store nucleus data
                nuclei_data.append({
                    'nucleus_id': i + 1,
                    'centroid': (cx, cy),
                    'nucleus_area_pixels': nucleus_area_pixels,
                    'nucleus_area_microns': nucleus_area_microns,
                    'nucleus_mean': nucleus_mean,
                    'nucleus_min': nucleus_min,
                    'nucleus_max': nucleus_max,
                    'mitosox_mean': mitosox_mean,
                    'mitosox_min': mitosox_min,
                    'mitosox_max': mitosox_max,
                    'mitosox_area_pixels': mitosox_area_pixels,
                    'mitosox_area_microns': mitosox_area_microns,
                    'contour': contour
                })
                
                # Draw nucleus number
                cv2.putText(result_img, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, nuclei_color, 1)

        # Add manually marked nuclei
        for i, (x, y) in enumerate(self.manual_nuclei):
            cv2.circle(result_img, (x, y), 15, nuclei_color, line_thickness)
            cv2.putText(result_img, str(len(contours) + i + 1), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, nuclei_color, 1)
            
            # Add manual nucleus data (simplified)
            nuclei_data.append({
                'nucleus_id': len(contours) + i + 1,
                'centroid': (x, y),
                'nucleus_area_pixels': 0,  # Manual nuclei don't have calculated area
                'nucleus_area_microns': 0,
                'nucleus_mean': 0,
                'nucleus_min': 0,
                'nucleus_max': 0,
                'mitosox_mean': 0,
                'mitosox_min': 0,
                'mitosox_max': 0,
                'mitosox_area_pixels': 0,
                'mitosox_area_microns': 0,
                'contour': None
            })

        return nuclei_data, result_img, len(contours) + len(self.manual_nuclei)

    def initialize_preview(self):
        """Initialize or update the preview matplotlib figure"""
        print("Starting initialize_preview...")
        if self.current_image is None:
            print("No current image available")
            return

        print(f"Processing image with shape: {self.current_image[0].shape}")
        # Process the image
        nuclei_data, result_img, count = self.process_image(
            self.current_image[0], self.current_image[1], self.current_image[2])
        print(f"Image processed. Count: {count}")

        # Update count label
        count_text = f"Nuclei count: {count} (Auto: {count - len(self.manual_nuclei)}, Manual: {len(self.manual_nuclei)})"
        self.count_label.config(text=count_text)

        # Create or update matplotlib figure
        if self.preview_fig is None:
            print("Creating new preview figure")
            # Create new figure
            self.preview_fig = plt.figure(figsize=(12, 8))

            # Create subplots - show original, nuclei, MitoSOX, and result
            gs = self.preview_fig.add_gridspec(2, 2)
            self.preview_axs = [
                self.preview_fig.add_subplot(gs[0, 0]),  # Original
                self.preview_fig.add_subplot(gs[0, 1]),  # Nuclei detection
                self.preview_fig.add_subplot(gs[1, 0]),  # MitoSOX channel
                self.preview_fig.add_subplot(gs[1, 1]),  # Result with measurements
            ]

            # Embed in tkinter
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            self.canvas = FigureCanvasTkAgg(self.preview_fig, master=self.canvas_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Add click event for adding missed nuclei
            self.canvas.mpl_connect('button_press_event', self.on_preview_click)

            # Add toolbar
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar_frame = ttk.Frame(self.canvas_frame)
            toolbar_frame.pack(fill=tk.X)
            toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
            toolbar.update()
            print("Preview figure created successfully")

        # Update the figure
        print("Updating preview figure")
        self.update_preview_figure(nuclei_data, result_img, count)
        print("Preview initialization complete")

    def update_preview_figure(self, nuclei_data, result_img, count):
        """Update the preview figure with the current image and processing results"""
        print("Starting update_preview_figure...")
        if self.preview_fig is None or self.preview_axs is None:
            print("Preview figure or axes not initialized")
            return

        print("Clearing axes...")
        # Clear all axes
        for ax in self.preview_axs:
            ax.clear()

        # Set tight layout to False to prevent automatic adjustment
        self.preview_fig.set_tight_layout(False)

        print("Displaying original image...")
        # Original image
        if len(self.current_image[0].shape) == 3:
            self.preview_axs[0].imshow(self.current_image[0])
        else:
            self.preview_axs[0].imshow(self.current_image[0], cmap='gray')
        self.preview_axs[0].set_title("Original Image")

        print("Displaying nuclei detection...")
        # Nuclei detection
        if len(self.current_image[0].shape) == 3:
            blue_channel = self.current_image[0][:, :, 2]
        else:
            blue_channel = self.current_image[0]
        self.preview_axs[1].imshow(blue_channel, cmap='Blues')
        self.preview_axs[1].set_title("Nuclei Channel")

        print("Displaying MitoSOX channel...")
        # MitoSOX channel
        if self.current_image[2] is not None:
            self.preview_axs[2].imshow(self.current_image[2], cmap='Reds')
        else:
            self.preview_axs[2].text(0.5, 0.5, 'No Red Channel', ha='center', va='center', transform=self.preview_axs[2].transAxes)
        self.preview_axs[2].set_title("MitoSOX Channel")

        print("Displaying result image...")
        # Result image with detections
        if len(result_img.shape) == 3:
            self.preview_axs[3].imshow(result_img)
        else:
            self.preview_axs[3].imshow(result_img, cmap='gray')
            
        self.preview_axs[3].set_title("Detection Results (Blue: Nuclei, Red: MitoSOX)")

        # Add title with count information
        title_text = f"Preview: {self.current_filename}\nNuclei: {count} (Auto: {count - len(self.manual_nuclei)}, Manual: {len(self.manual_nuclei)})"
        self.preview_fig.suptitle(title_text, fontsize=12)

        # Add instruction about manual marking
        self.preview_fig.text(0.5, 0.01,
                              "Click on missed nuclei in the Detection Result panel to mark them manually",
                              ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

        print("Adjusting layout...")
        # Use fixed spacing instead of tight_layout
        self.preview_fig.subplots_adjust(
            left=0.05, right=0.95, bottom=0.1, top=0.9,
            wspace=0.2, hspace=0.3
        )

        print("Drawing canvas...")
        # Draw the updated figure
        self.canvas.draw()
        print("Preview figure update complete")

    def update_preview(self):
        """Update the preview with current parameters"""
        if self.current_image is None:
            return

        # Process the image with current parameters
        nuclei_data, result_img, count = self.process_image(
            self.current_image[0], self.current_image[1], self.current_image[2])

        # Update the preview figure
        self.update_preview_figure(nuclei_data, result_img, count)

        # Update count label
        count_text = f"Nuclei count: {count} (Auto: {count - len(self.manual_nuclei)}, Manual: {len(self.manual_nuclei)})"
        self.count_label.config(text=count_text)

    def on_preview_click(self, event):
        """Handle click events to add missed nuclei"""
        if event.inaxes == self.preview_axs[3]:  # Click on result panel
            x, y = int(event.xdata), int(event.ydata)

            # Check if within image bounds
            if 0 <= x < self.current_image[0].shape[1] and 0 <= y < self.current_image[0].shape[0]:
                # Add to manual nuclei list
                self.manual_nuclei.append((x, y))
                print(f"Manual nucleus added at ({x}, {y})")

                # Update preview
                self.update_preview()

    def save_current_image(self):
        """Save results for the current image"""
        if self.current_image is None or self.current_filename is None:
            return False

        # Process image with current parameters
        nuclei_data, result_img, count = self.process_image(
            self.current_image[0], self.current_image[1], self.current_image[2])

        # Save result image
        output_path = os.path.join(self.output_dir, f"result_{self.current_filename}")
        io.imsave(output_path, result_img)

        # Store nucleus data for this image
        self.nuclei_data[self.current_filename] = nuclei_data

        # Create detailed data for Excel export
        detailed_data = []
        for nucleus in nuclei_data:
            detailed_data.append({
                'Filename': self.current_filename,
                'Nucleus_ID': nucleus['nucleus_id'],
                'Nucleus_Area_Pixels': nucleus['nucleus_area_pixels'],
                'Nucleus_Area_Microns': nucleus['nucleus_area_microns'],
                'Nucleus_Mean_Intensity': nucleus['nucleus_mean'],
                'Nucleus_Min_Intensity': nucleus['nucleus_min'],
                'Nucleus_Max_Intensity': nucleus['nucleus_max'],
                'MitoSOX_Mean_Intensity': nucleus['mitosox_mean'],
                'MitoSOX_Min_Intensity': nucleus['mitosox_min'],
                'MitoSOX_Max_Intensity': nucleus['mitosox_max'],
                'MitoSOX_Area_Pixels': nucleus['mitosox_area_pixels'],
                'MitoSOX_Area_Microns': nucleus['mitosox_area_microns']
            })

        # Update or add to total results
        existing_entry = next((item for item in self.total_results
                               if item['Filename'] == self.current_filename), None)
        if existing_entry:
            existing_entry.update({
                'Total_Nuclei_Count': count,
                'Auto_Nuclei_Count': count - len(self.manual_nuclei),
                'Manual_Nuclei_Count': len(self.manual_nuclei)
            })
        else:
            self.total_results.append({
                'Filename': self.current_filename,
                'Total_Nuclei_Count': count,
                'Auto_Nuclei_Count': count - len(self.manual_nuclei),
                'Manual_Nuclei_Count': len(self.manual_nuclei)
            })

        # Save detailed nucleus data to CSV
        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_csv_path = os.path.join(self.output_dir, f"nucleus_data_{self.current_filename.split('.')[0]}.csv")
            detailed_df.to_csv(detailed_csv_path, index=False)

        # Update stored manual nuclei
        if self.manual_nuclei:
            self.stored_manual_nuclei[self.current_filename] = self.manual_nuclei.copy()

        # Save custom parameters if modified
        if self.param_modified:
            self.image_params[self.current_filename] = self.params.copy()

        # Prepare success message
        success_msg = f"Results saved for {self.current_filename}: {count} nuclei detected with MitoSOX analysis"
        messagebox.showinfo("Saved", success_msg)
        return True

    def move_to_next_image(self):
        """Save current image and move to the next one"""
        if self.save_current_image():
            # Move to next image
            self.current_file_index += 1
            if self.current_file_index < len(self.tiff_files):
                self.load_current_image()
                return True
            else:
                # No more images
                self.finalize_processing()
                return False
        return False

    def move_to_prev_image(self):
        """Save current image and move to the previous one"""
        if self.save_current_image():
            # Move to previous image
            if self.current_file_index > 0:
                self.current_file_index -= 1
                self.load_current_image()
                return True
        return False

    def process_remaining(self):
        """Process all remaining images without further user interaction"""
        response = messagebox.askyesno(
            "Process Remaining",
            f"Process the remaining {len(self.tiff_files) - self.current_file_index} images with current settings?"
        )

        if response:
            # Save current image
            self.save_current_image()

            # Process remaining images
            start_index = self.current_file_index + 1
            for i in range(start_index, len(self.tiff_files)):
                self.current_file_index = i
                if self.load_current_image():
                    self.save_current_image()
                    print(f"Auto-processed {self.current_filename}")

            self.finalize_processing()

    def finalize_processing(self):
        """Finalize processing and save summary Excel"""
        if not self.total_results:
            messagebox.showinfo("No Results", "No images were processed.")
            return

        # Create comprehensive Excel report
        excel_path = os.path.join(self.output_dir, "mitosox_analysis_report.xlsx")
        
        # Prepare summary data
        summary_data = []
        all_nucleus_data = []
        
        for result in self.total_results:
            filename = result['Filename']
            summary_data.append({
                'Filename': filename,
                'Total_Nuclei_Count': result['Total_Nuclei_Count'],
                'Auto_Nuclei_Count': result['Auto_Nuclei_Count'],
                'Manual_Nuclei_Count': result['Manual_Nuclei_Count']
            })
            
            # Add detailed nucleus data for this file
            if filename in self.nuclei_data:
                for nucleus in self.nuclei_data[filename]:
                    all_nucleus_data.append({
                        'Filename': filename,
                        'Nucleus_ID': nucleus['nucleus_id'],
                        'Nucleus_Area_Pixels': nucleus['nucleus_area_pixels'],
                        'Nucleus_Area_Microns': nucleus['nucleus_area_microns'],
                        'Nucleus_Mean_Intensity': nucleus['nucleus_mean'],
                        'Nucleus_Min_Intensity': nucleus['nucleus_min'],
                        'Nucleus_Max_Intensity': nucleus['nucleus_max'],
                        'MitoSOX_Mean_Intensity': nucleus['mitosox_mean'],
                        'MitoSOX_Min_Intensity': nucleus['mitosox_min'],
                        'MitoSOX_Max_Intensity': nucleus['mitosox_max'],
                        'MitoSOX_Area_Pixels': nucleus['mitosox_area_pixels'],
                        'MitoSOX_Area_Microns': nucleus['mitosox_area_microns']
                    })

        # Save to Excel with multiple sheets
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed nucleus data sheet
            if all_nucleus_data:
                pd.DataFrame(all_nucleus_data).to_excel(writer, sheet_name='Nucleus Details', index=False)
            
            # Format the sheets
            for sheet in writer.sheets.values():
                for column in sheet.columns:
                    max_length = max(len(str(cell.value)) for cell in column)
                    sheet.column_dimensions[column[0].column_letter].width = min(max_length + 2, 30)

        print(f"Excel report saved to {excel_path}")
        
        # Show completion message
        total_nuclei = sum(result['Total_Nuclei_Count'] for result in summary_data)
        message = f"""MitoSOX Analysis Complete!

Summary:
- Total images processed: {len(summary_data)}
- Total nuclei analyzed: {total_nuclei}

Results saved to: {self.output_dir}
Excel report: mitosox_analysis_report.xlsx

Each nucleus row includes:
• Nucleus area and intensity statistics
• MitoSOX signal intensity statistics
• All measurements in both pixels and microns"""
        
        messagebox.showinfo("Processing Complete", message)
        
        print("Processing completed. Closing application.")
        self.root.after(200, self.close_application)

    def close_application(self):
        """Close the application safely"""
        self.root.quit()
        self.root.destroy()
        sys.exit(0)


# Run the application
if __name__ == "__main__":
    app = MitoSOXAnalyzer()
