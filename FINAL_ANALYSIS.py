#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script:
# 0. Let the use select a folder with TIFF images
# 1. Identifies blue stains in TIFF images [nuclei] --> count them and let you control the threshold live
# 2. Counts the total number of red dots and their area in microns
# 3. Counts the total number of green dots (only when checking the checkbox at the beginning of the run) and calculate their area
# 4. Extract: processed images, each image data (count of dots and area of each dot) and a summary Excel sheet with:
# File name, Total blue stains, total red dots, total red area, red dots/blue stains, total red area/blue stains,
# avg red dot size, total green dots, green dot/blue stains total green area, total green area/blue stains

# Error handling: all errors are printed to console. If no TIFF file exist in the folder a message will prompt
# Custom controls: you may change the thresholds per image easily by using the slider on the GUI. By default, changes will only affect the current image.
# If you wish to totally change the values for all images in this batch, click 'apply to all'

####################################################

# Required imports
import os
import sys
import traceback
import numpy as np
import pandas as pd
import cv2
from skimage import io, morphology, measure, segmentation, filters, feature
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from tkinter import ttk
import matplotlib
import warnings

# Ignore openpyxl warning about default styles
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl.styles')

matplotlib.use('TkAgg')


class NucleiCounter:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DAPI Nuclei & Dot Counter")
        
        # Set up exception handling
        sys.excepthook = self.handle_exception
        
        # Start with a small window size for the initial dialog
        self.root.geometry("400x150")
        self.root.update_idletasks()  # To ensure geometry is applied
        
        # Center the initial dialog
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 400) // 2
        y = (screen_height - 150) // 2
        self.root.geometry(f"400x150+{x}+{y}")
        
        # Ask user if they want to analyze green dots
        self.analyze_green_dots = False
        self.initial_frame = ttk.Frame(self.root, padding=20)
        self.initial_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(self.initial_frame, text="Staining Analysis Options", font=("Arial", 14, "bold")).pack(pady=(0, 15))
        
        self.green_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.initial_frame, text="Analyze green dots in addition to red dots", 
                         variable=self.green_var).pack(pady=5, anchor=tk.W)
        
        ttk.Button(self.initial_frame, text="Start Analysis", command=self.start_analysis).pack(pady=15)
        
        # Wait for user to click the Start Analysis button
        # Don't use wait_variable here as it causes issues
        
        # Run the main loop
        self.root.mainloop()
        
    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions by showing a message box with the error details"""
        # Format the exception information
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        
        # Display error in console
        print("ERROR:", error_msg)
        
        # Show in message box
        messagebox.showerror("Application Error", 
                            f"An error occurred:\n\n{exc_value}\n\nCheck console for full details.")
        
        # Call the default exception handler
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        
    def start_analysis(self):
        """Start the main analysis after initial options are selected"""
        try:
            print("Starting analysis...")
            self.analyze_green_dots = self.green_var.get()
            print(f"Analyze green dots: {self.analyze_green_dots}")
            
            # Destroy initial dialog
            self.initial_frame.destroy()
            
            # Reset window size for main application
            self.root.geometry("1200x800")
            
            # Default parameters
            self.default_params = {
                'blue_threshold': 65,  # Threshold for blue channel
                'min_size': 153,  # Minimum nucleus size in pixels
                'max_size': 10000,  # Maximum nucleus size in pixels
                'dilation_size': 5,  # Size of dilation kernel to connect fragments
                'closing_size': 4,  # Size of closing kernel
                'distance_threshold': 15,  # Distance threshold for connecting fragments
                'red_threshold': 100,  # Threshold for red channel
                'red_min_size': 5,  # Minimum red dot size in pixels
                'red_max_size': 500  # Maximum red dot size in pixels
            }
            
            # Add green dot parameters if needed
            if self.analyze_green_dots:
                print("Adding green dot parameters")
                self.default_params.update({
                    'green_threshold': 100,  # Threshold for green channel
                    'green_min_size': 5,  # Minimum green dot size in pixels
                    'green_max_size': 500  # Maximum green dot size in pixels
                })
            
            # Micron conversion factor
            self.MICRON_CONVERSION = 5.7273  # 1 micron = 5.7273 pixels
            
            # For storing red dots
            self.red_dots = []
            self.stored_red_dots = {}
            
            # For storing green dots if needed
            if self.analyze_green_dots:
                print("Initializing green dot storage")
                self.green_dots = []
                self.stored_green_dots = {}

            # Current parameters (will be image-specific)
            self.params = self.default_params.copy()

            # Store image-specific parameters
            self.image_params = {}

            # UI elements
            self.preview_fig = None
            self.preview_axs = None
            self.binary_img = None
            self.result_img = None
            self.param_frame = None
            self.sliders = {}
            self.param_modified = False  # Flag to track if parameters were modified for current image

            # Data
            self.current_image = None
            self.current_filename = None
            self.output_dir = None
            self.total_results = []
            self.current_file_index = 0
            self.input_dir = None
            self.tiff_files = []

            # User-added nuclei storage
            self.manual_nuclei = []
            self.stored_manual_nuclei = {}

            print("Creating main GUI...")
            # Create main GUI
            self.create_main_gui()
            print("Main GUI created successfully")

        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Error in start_analysis: {e}\n{error_details}")
            messagebox.showerror("Error Starting Analysis", 
                                f"An error occurred while starting the analysis: {e}\n\nSee console for details.")
            # Recreate the initial dialog if there was an error
            self.recreate_initial_dialog()
        
    def recreate_initial_dialog(self):
        """Recreate the initial dialog if there was an error in start_analysis"""
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
            
        # Reset window size
        self.root.geometry("400x150")
        
        # Recreate the initial dialog
        self.initial_frame = ttk.Frame(self.root, padding=20)
        self.initial_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(self.initial_frame, text="Staining Analysis Options", font=("Arial", 14, "bold")).pack(pady=(0, 15))
        
        self.green_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.initial_frame, text="Analyze green dots in addition to red dots", 
                         variable=self.green_var).pack(pady=5, anchor=tk.W)
        
        ttk.Button(self.initial_frame, text="Start Analysis", command=self.start_analysis).pack(pady=15)
        
    def create_main_gui(self):
        """Create the main GUI with all necessary frames"""
        # Set window size
        self.root.geometry("1200x800")  # Increase window size

        # Create main frames
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create header with file selection
        header_frame = ttk.Frame(self.main_frame, padding=5)
        header_frame.pack(fill=tk.X, pady=5)

        title_text = "DAPI-Stained Nuclei & Dot Counter"
        if self.analyze_green_dots:
            title_text += " (Red + Green)"
        else:
            title_text += " (Red)"
            
        ttk.Label(header_frame, text=title_text, font=("Arial", 16, "bold")).pack(side=tk.LEFT, padx=5)

        select_btn = ttk.Button(header_frame, text="Select Folder with TIFF Images",
                                command=self.select_and_load_folder)
        select_btn.pack(side=tk.RIGHT, padx=5)

        # Create main content frame
        content_frame = ttk.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Left frame for parameters - set fixed width
        self.param_frame = ttk.LabelFrame(content_frame, text="Parameters", padding=10, width=350)
        self.param_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        self.param_frame.pack_propagate(False)  # Prevent frame from shrinking

        # Initialize param_status for tracking parameter changes
        self.param_status = ttk.Label(self.param_frame, text="Using default parameters")
        self.param_status.pack(anchor="w", pady=5)

        # Right frame for previews and navigation - ensure it takes the remaining space
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
        
        # Navigation buttons - Above the image preview
        buttons_frame = ttk.Frame(preview_frame)
        buttons_frame.pack(fill=tk.X, pady=10)

        # Use grid layout for buttons to ensure they're all visible
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
        
        # Status indicator with larger font for better visibility
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
        self.output_dir = os.path.join(input_dir, "nuclei_results")
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
            
            # Make buttons more visible
            self.save_btn.config(width=20)
            self.next_btn.config(width=20)
            
            # Update UI status to make it clear the app is ready
            self.ui_status.config(text="Image loaded - Ready to process!", foreground="green")
            
            # Create parameter sliders
            self.create_parameter_sliders()
        else:
            messagebox.showerror("Error", "Failed to load the first image.")
            print("Failed to load the first image.")
            self.ui_status.config(text="Error loading image. Please try again.", foreground="red")

    def create_parameter_sliders(self):
        """Create sliders for each parameter"""
        # Clear existing widgets in the parameter frame, except info_frame and param_status
        for widget in self.param_frame.winfo_children():
            if widget != self.param_status:  # Keep param_status
                widget.destroy()

        # Add explanatory text
        info_frame = ttk.Frame(self.param_frame, borderwidth=2, relief="groove", padding=10)
        info_frame.pack(fill=tk.X, pady=10)

        ttk.Label(info_frame, text="Image-Specific Parameters", font=("Arial", 12, "bold"), foreground="blue").pack(
            anchor="w")

        ttk.Label(
            info_frame,
            text="• Parameters you adjust will ONLY affect the current image\n"
                 "• Each image maintains its own settings\n"
                 "• Use 'Apply to All Images' to make current settings the new defaults",
            font=("Arial", 10),
            justify="left"
        ).pack(anchor="w", pady=5)

        # Re-pack the param_status for current image
        self.param_status.pack(anchor="w", pady=5)

        # Parameter ranges and step sizes
        param_configs = {
            'blue_threshold': (0, 255, 1),
            'min_size': (10, 500, 10),
            'max_size': (500, 10000, 100),
            'dilation_size': (0, 20, 1),
            'closing_size': (0, 20, 1),
            'distance_threshold': (1, 30, 1),
            'red_threshold': (0, 255, 1),
            'red_min_size': (1, 500, 10),
            'red_max_size': (1, 500, 10)
        }
        
        # Add green dot parameters if needed
        if self.analyze_green_dots:
            param_configs.update({
                'green_threshold': (0, 255, 1),
                'green_min_size': (1, 500, 10),
                'green_max_size': (1, 500, 10)
            })

        # Create frame for sliders
        sliders_frame = ttk.Frame(self.param_frame)
        sliders_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.sliders = {}
        row = 0

        # Create sliders
        for param, value in self.params.items():
            label = ttk.Label(sliders_frame, text=param.replace('_', ' ').title())
            label.grid(row=row, column=0, sticky=tk.W, pady=5)

            min_val, max_val, step = param_configs[param]

            var = tk.IntVar(value=value)
            slider = ttk.Scale(
                sliders_frame,
                from_=min_val,
                to=max_val,
                variable=var,
                orient=tk.HORIZONTAL,
                length=200,
                command=lambda v, p=param, var=var: self.update_param(p, int(float(var.get())))
            )
            slider.grid(row=row, column=1, padx=5, pady=5)

            value_label = ttk.Label(sliders_frame, text=str(value))
            value_label.grid(row=row, column=2, padx=5)

            # Add reset to default button for each parameter
            reset_btn = ttk.Button(
                sliders_frame,
                text="Reset",
                width=6,
                command=lambda p=param: self.reset_param_to_default(p)
            )
            reset_btn.grid(row=row, column=3, padx=5, pady=5)

            self.sliders[param] = (var, value_label, reset_btn)
            row += 1

            # Set initial value and trace changes
            var.trace_add("write", lambda *args, p=param, v=var, l=value_label:
            l.config(text=str(int(float(v.get())))))

        # Parameter action buttons
        actions_frame = ttk.Frame(self.param_frame)
        actions_frame.pack(fill=tk.X, pady=10)

        # Button to reset all parameters to default
        reset_all_btn = ttk.Button(
            actions_frame,
            text="Reset All to Default",
            command=self.reset_all_params
        )
        reset_all_btn.pack(side=tk.LEFT, padx=5)

        # Button to apply current parameters to all images
        apply_all_btn = ttk.Button(
            actions_frame,
            text="Apply to All Images",
            command=self.apply_params_to_all
        )
        apply_all_btn.pack(side=tk.LEFT, padx=5)

        # Add a button to clear manual nuclei
        clear_manual_btn = ttk.Button(self.param_frame, text="Clear Manual Markers", command=self.clear_manual_nuclei)
        clear_manual_btn.pack(fill=tk.X, pady=10)

    def update_param(self, param, value):
        """Update parameter value and refresh preview"""
        # Save previous value to check if changed
        prev_value = self.params[param]

        # Update parameter
        self.params[param] = value

        # Mark as modified if changed from default
        if value != self.default_params[param]:
            self.param_modified = True
            # Store image-specific parameters
            self.image_params[self.current_filename] = self.params.copy()

            # Update status text
            if hasattr(self, 'param_status'):
                self.param_status.config(text="* This image has custom parameters *")
        elif prev_value != value:
            # Check if all parameters now match defaults
            all_default = all(self.params[p] == self.default_params[p] for p in self.params)
            if all_default:
                # If using all defaults, remove from image_params if present
                if self.current_filename in self.image_params:
                    del self.image_params[self.current_filename]
                self.param_modified = False

                # Update status text
                if hasattr(self, 'param_status'):
                    self.param_status.config(text="Using default parameters")

        # Update preview
        self.update_preview()

    def reset_param_to_default(self, param):
        """Reset a specific parameter to its default value"""
        default_value = self.default_params[param]

        # Update slider
        slider_var, _, _ = self.sliders[param]
        slider_var.set(default_value)

        # This will trigger update_param through the trace

    def reset_all_params(self):
        """Reset all parameters to default values"""
        for param, (slider_var, _, _) in self.sliders.items():
            slider_var.set(self.default_params[param])

        # Update status
        if self.current_filename in self.image_params:
            del self.image_params[self.current_filename]

        self.param_modified = False
        self.param_status.config(text="Using default parameters")

    def apply_params_to_all(self):
        """Apply current parameters to all images"""
        # Ask for confirmation
        confirm = messagebox.askyesno(
            "Apply to All Images",
            "Are you sure you want to apply the current parameters to ALL images?\n"
            "This will override any custom settings for individual images."
        )

        if confirm:
            # Update default parameters
            self.default_params = self.params.copy()

            # Clear all image-specific parameters
            self.image_params = {}

            # Update status
            self.param_modified = False
            self.param_status.config(text="Using default parameters (applied to all)")

            messagebox.showinfo(
                "Parameters Applied",
                "Current parameters have been set as the new default for all images."
            )

    def load_current_image(self):
        """Load the current image based on index"""
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
            elif len(image.shape) == 3:  # RGB or RGBA
                print(f"Detected color image with {image.shape[2]} channels")
                if image.shape[2] >= 3:  # At least 3 channels (RGB)
                    blue_channel = image[:, :, 2]  # blue channel is index 2 in RGB
                    print("Extracted blue channel (index 2)")
                else:
                    raise ValueError(f"Unexpected number of channels: {image.shape[2]}")
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")

            self.current_image = (image, blue_channel)

            # Debug information about loaded image
            print(f"Current image dimensions: {image.shape}")
            print(f"Blue channel dimensions: {blue_channel.shape}")
            print(f"Blue channel min/max: {blue_channel.min()}/{blue_channel.max()}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image {filename}: {e}")
            print(f"Error loading image: {str(e)}")
            return False

        # Retrieve any stored manual nuclei for this image
        if filename in self.stored_manual_nuclei:
            self.manual_nuclei = self.stored_manual_nuclei[filename].copy()
        else:
            self.manual_nuclei = []
            
        # Retrieve any stored red dots for this image
        if filename in self.stored_red_dots:
            self.red_dots = self.stored_red_dots[filename].copy()
        else:
            self.red_dots = []
            
        # Retrieve any stored green dots for this image if applicable
        if self.analyze_green_dots:
            if filename in self.stored_green_dots:
                self.green_dots = self.stored_green_dots[filename].copy()
            else:
                self.green_dots = []

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
            for param, (slider_var, value_label, _) in self.sliders.items():
                slider_var.set(self.params[param])

        # Update navigation status
        current_idx = self.current_file_index + 1
        self.status_label.config(text=f"Image {current_idx} of {len(self.tiff_files)}: {filename}")

        # Update preview
        self.initialize_preview()

        # Update navigation buttons
        self.prev_btn.config(state=tk.NORMAL if self.current_file_index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_file_index < len(self.tiff_files) - 1 else tk.DISABLED)

        return True

    def initialize_preview(self):
        """Initialize or update the preview matplotlib figure"""
        if self.current_image is None:
            return

        # Process the image
        binary, nuclei_mask, result_img, count = self.process_image(
            self.current_image[0], self.current_image[1])

        # Update count label
        count_text = f"Nuclei count: {count} (Auto: {count - len(self.manual_nuclei)}, Manual: {len(self.manual_nuclei)}) | Red dots: {len(self.red_dots)}"
        if self.analyze_green_dots:
            count_text += f" | Green dots: {len(self.green_dots)}"
        self.count_label.config(text=count_text)

        # Create or update matplotlib figure
        if self.preview_fig is None:
            # Create new figure
            self.preview_fig = plt.figure(figsize=(10, 8))

            # Create subplots - only show original and result image
            gs = self.preview_fig.add_gridspec(1, 2)
            self.preview_axs = [
                self.preview_fig.add_subplot(gs[0, 0]),  # Original
                self.preview_fig.add_subplot(gs[0, 1]),  # Result with detections
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

        # Update the figure
        self.update_preview_figure(binary, nuclei_mask, result_img, count)

    def update_preview_figure(self, binary, nuclei_mask, result_img, count):
        """Update the preview figure with the current image and processing results"""
        if self.preview_fig is None or self.preview_axs is None:
            return

        # Clear all axes
        for ax in self.preview_axs:
            ax.clear()

        # Set tight layout to False to prevent automatic adjustment
        self.preview_fig.set_tight_layout(False)

        # Original image
        if len(self.current_image[0].shape) == 3:
            self.preview_axs[0].imshow(self.current_image[0])
        else:
            self.preview_axs[0].imshow(self.current_image[0], cmap='gray')
        self.preview_axs[0].set_title("Original Image")

        # Result image with detections
        if len(result_img.shape) == 3:
            # Handle color conversion if needed
            if np.array_equal(result_img[:, :, 0], self.current_image[0][:, :, 2]) and \
                    np.array_equal(result_img[:, :, 2], self.current_image[0][:, :, 0]):
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            self.preview_axs[1].imshow(result_img)
        else:
            self.preview_axs[1].imshow(result_img, cmap='gray')
            
        # Set title with color legend
        title = "Detection Results"
        if self.analyze_green_dots:
            title += " (Yellow: Blue nuclei & Red dots, Magenta: Green dots)"
        else:
            title += " (Yellow outlines: Blue nuclei & Red dots)"
        self.preview_axs[1].set_title(title)

        # Add title with count information
        title_text = f"Preview: {self.current_filename}\nNuclei: {count} (Auto: {count - len(self.manual_nuclei)}, Manual: {len(self.manual_nuclei)}) | Red Dots: {len(self.red_dots)}"
        if self.analyze_green_dots:
            title_text += f" | Green Dots: {len(self.green_dots)}"
        self.preview_fig.suptitle(title_text, fontsize=12)

        # Add instruction about manual marking
        self.preview_fig.text(0.5, 0.01,
                              "Click on missed nuclei in the Detection Result panel to mark them manually",
                              ha='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

        # Use fixed spacing instead of tight_layout
        self.preview_fig.subplots_adjust(
            left=0.05, right=0.95, bottom=0.1, top=0.9,
            wspace=0.2, hspace=0.3
        )

        # Draw the updated figure
        self.canvas.draw()

    def update_preview(self):
        """Update the preview with current parameters"""
        if self.current_image is None:
            return

        # Process the image with current parameters
        binary, nuclei_mask, result_img, count = self.process_image(
            self.current_image[0], self.current_image[1])

        # Update the preview figure
        self.update_preview_figure(binary, nuclei_mask, result_img, count)

        # Update count label
        count_text = f"Nuclei count: {count} (Auto: {count - len(self.manual_nuclei)}, Manual: {len(self.manual_nuclei)}) | Red dots: {len(self.red_dots)}"
        if self.analyze_green_dots:
            count_text += f" | Green dots: {len(self.green_dots)}"
        self.count_label.config(text=count_text)

    def on_preview_click(self, event):
        """Handle click events to add missed nuclei"""
        if event.inaxes == self.preview_axs[1]:  # Click on result panel
            x, y = int(event.xdata), int(event.ydata)

            # Check if within image bounds
            if 0 <= x < self.current_image[0].shape[1] and 0 <= y < self.current_image[0].shape[0]:
                # Add to manual nuclei list
                self.manual_nuclei.append((x, y))
                print(f"Manual nucleus added at ({x}, {y})")

                # Update preview
                self.update_preview()

    def clear_manual_nuclei(self):
        """Clear all manually added nuclei markers"""
        self.manual_nuclei = []
        self.update_preview()

    def process_image(self, image, blue_channel):
        """Process image to detect nuclei and red dots"""
        # Apply Gaussian blur to reduce noise
        blurred_blue = cv2.GaussianBlur(blue_channel, (5, 5), 0)

        # Apply threshold to get binary image
        _, binary = cv2.threshold(blurred_blue, self.params['blue_threshold'], 255, cv2.THRESH_BINARY)

        # Apply morphological operations to connect fragments
        if self.params['closing_size'] > 0:
            closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                       (self.params['closing_size'], self.params['closing_size']))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, closing_kernel)

        if self.params['dilation_size'] > 0:
            dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                        (self.params['dilation_size'], self.params['dilation_size']))
            binary = cv2.dilate(binary, dilation_kernel, iterations=1)
            binary = cv2.erode(binary, dilation_kernel, iterations=1)

        # Label connected components
        labels = measure.label(binary)
        regions = measure.regionprops(labels)

        # Filter by size and create a mask for valid nuclei
        valid_labels = [region.label for region in regions
                        if self.params['min_size'] <= region.area <= self.params['max_size']]

        nuclei_mask = np.zeros_like(binary)
        for label in valid_labels:
            nuclei_mask[labels == label] = 255

        # Create a result image with outlines
        if len(image.shape) == 2:  # Grayscale
            result_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result_img = image.copy()

        # Find contours of nuclei
        contours, _ = cv2.findContours(nuclei_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours and number them - using thinner lines (30% of original 2px thickness)
        nuclei_line_thickness = int(2 * 0.3) if int(2 * 0.3) >= 1 else 1  # Ensure minimum thickness of 1
        for i, contour in enumerate(contours):
            cv2.drawContours(result_img, [contour], -1, (0, 255, 255), nuclei_line_thickness)
            # Draw nucleus number only (no size data)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(result_img, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

        # Add manually marked nuclei
        for i, (x, y) in enumerate(self.manual_nuclei):
            # Draw a circle for each manually added nucleus - thinner line
            cv2.circle(result_img, (x, y), 15, (0, 255, 255), nuclei_line_thickness)
            # Label with number continuing from automatic detections
            cv2.putText(result_img, str(len(contours) + i + 1), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

        # Process red channel if the image is RGB
        red_dots = []
        green_dots = []
        total_green_area_microns = 0  # Initialize total green area
        
        if len(image.shape) == 3 and image.shape[2] >= 3:
            # Extract red channel
            red_channel = image[:, :, 0]  # red channel is index 0 in RGB
            
            # Apply Gaussian blur
            blurred_red = cv2.GaussianBlur(red_channel, (5, 5), 0)
            
            # Apply threshold to get binary image of red dots
            _, red_binary = cv2.threshold(blurred_red, self.params['red_threshold'], 255, cv2.THRESH_BINARY)
            
            # Label connected components
            red_labels = measure.label(red_binary)
            red_regions = measure.regionprops(red_labels)
            
            # Filter red dots by size
            valid_red_labels = [region.label for region in red_regions
                             if self.params['red_min_size'] <= region.area <= self.params['red_max_size']]
            
            # Create a mask for valid red dots
            red_mask = np.zeros_like(red_binary)
            for label in valid_red_labels:
                red_mask[red_labels == label] = 255
            
            # Find contours of red dots
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate thinner line thickness for dots (30% of original 2px)
            dot_line_thickness = int(2 * 0.3) if int(2 * 0.3) >= 1 else 1  # Ensure minimum thickness of 1
            
            # Draw contours on result image
            for i, contour in enumerate(red_contours):
                # Calculate size in microns
                area_pixels = cv2.contourArea(contour)
                area_microns = area_pixels / (self.MICRON_CONVERSION * self.MICRON_CONVERSION)
                
                # Store the dot data
                red_dots.append({
                    'id': i + 1,
                    'area_pixels': area_pixels,
                    'area_microns': area_microns,
                    'contour': contour
                })
                
                # Draw yellow contour for red dots with thinner line
                # Yellow color in BGR is (0, 255, 255)
                cv2.drawContours(result_img, [contour], -1, (0, 255, 255), dot_line_thickness)
            
            # Always extract green channel for total area, regardless of analyze_green_dots setting
            green_channel = image[:, :, 1]  # green channel is index 1 in RGB
            
            # Default threshold for green channel if not analyzing green dots
            green_threshold = self.params.get('green_threshold', 100)
            
            # Apply Gaussian blur
            blurred_green = cv2.GaussianBlur(green_channel, (5, 5), 0)
            
            # Apply threshold to get binary image of green areas
            _, green_binary = cv2.threshold(blurred_green, green_threshold, 255, cv2.THRESH_BINARY)
            
            # Calculate total green area in microns (all pixels above threshold)
            total_green_pixels = np.sum(green_binary > 0)
            total_green_area_microns = total_green_pixels / (self.MICRON_CONVERSION * self.MICRON_CONVERSION)
            
            # Process green dots if requested (individual dots analysis)
            if self.analyze_green_dots:
                # Label connected components for individual dots
                green_labels = measure.label(green_binary)
                green_regions = measure.regionprops(green_labels)
                
                # Filter green dots by size
                valid_green_labels = [region.label for region in green_regions
                                 if self.params['green_min_size'] <= region.area <= self.params['green_max_size']]
                
                # Create a mask for valid green dots
                green_mask = np.zeros_like(green_binary)
                for label in valid_green_labels:
                    green_mask[green_labels == label] = 255
                
                # Find contours of green dots
                green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours on result image
                for i, contour in enumerate(green_contours):
                    # Calculate size in microns
                    area_pixels = cv2.contourArea(contour)
                    area_microns = area_pixels / (self.MICRON_CONVERSION * self.MICRON_CONVERSION)
                    
                    # Store the dot data
                    green_dots.append({
                        'id': i + 1,
                        'area_pixels': area_pixels,
                        'area_microns': area_microns,
                        'contour': contour
                    })
                    
                    # Draw magenta contour for green dots with thinner line
                    # Magenta color in BGR is (255, 0, 255)
                    cv2.drawContours(result_img, [contour], -1, (255, 0, 255), dot_line_thickness)
        
        # Save detected dots
        self.red_dots = red_dots
        if self.analyze_green_dots:
            self.green_dots = green_dots
            
        # Store total green area regardless of analyze_green_dots setting
        self.total_green_area_microns = total_green_area_microns

        return binary, nuclei_mask, result_img, len(contours) + len(self.manual_nuclei)

    def save_current_image(self):
        """Save results for the current image"""
        if self.current_image is None or self.current_filename is None:
            return False

        # Process image with current parameters
        binary, nuclei_mask, result_img, count = self.process_image(
            self.current_image[0], self.current_image[1])

        # Save result image
        output_path = os.path.join(self.output_dir, f"result_{self.current_filename}")
        io.imsave(output_path, result_img)

        # Store results for nuclei
        result_entry = {
            'Filename': self.current_filename,
            'Auto_Nuclei_Count': count - len(self.manual_nuclei),
            'Manual_Nuclei_Count': len(self.manual_nuclei),
            'Total_Nuclei_Count': count,
            'Red_Dot_Count': len(self.red_dots),
            'Total_Green_Area_Microns': self.total_green_area_microns,  # Always store total green area
            'Has_Custom_Params': self.current_filename in self.image_params
        }
        
        # Add green dot count if applicable
        if self.analyze_green_dots:
            result_entry['Green_Dot_Count'] = len(self.green_dots)
        
        # Update or add to total results
        existing_entry = next((item for item in self.total_results
                               if item['Filename'] == self.current_filename), None)
        if existing_entry:
            existing_entry.update(result_entry)
        else:
            self.total_results.append(result_entry)

        # Update stored manual nuclei
        if self.manual_nuclei:
            self.stored_manual_nuclei[self.current_filename] = self.manual_nuclei.copy()
            
        # Update stored red dots
        if self.red_dots:
            self.stored_red_dots[self.current_filename] = self.red_dots.copy()
            
            # Save red dot data to separate CSV
            red_dot_data = []
            for dot in self.red_dots:
                red_dot_data.append({
                    'Filename': self.current_filename,
                    'Dot_ID': dot['id'],
                    'Area_Pixels': dot['area_pixels'],
                    'Area_Microns': dot['area_microns']
                })
            
            if red_dot_data:
                red_dot_df = pd.DataFrame(red_dot_data)
                red_dots_csv_path = os.path.join(self.output_dir, f"red_dots_{self.current_filename.split('.')[0]}.csv")
                red_dot_df.to_csv(red_dots_csv_path, index=False)
        
        # Update stored green dots and save data if applicable
        if self.analyze_green_dots and self.green_dots:
            self.stored_green_dots[self.current_filename] = self.green_dots.copy()
            
            # Save green dot data to separate CSV
            green_dot_data = []
            for dot in self.green_dots:
                green_dot_data.append({
                    'Filename': self.current_filename,
                    'Dot_ID': dot['id'],
                    'Area_Pixels': dot['area_pixels'],
                    'Area_Microns': dot['area_microns']
                })
            
            if green_dot_data:
                green_dot_df = pd.DataFrame(green_dot_data)
                green_dots_csv_path = os.path.join(self.output_dir, f"green_dots_{self.current_filename.split('.')[0]}.csv")
                green_dot_df.to_csv(green_dots_csv_path, index=False)

        # Save custom parameters if modified
        if self.param_modified:
            self.image_params[self.current_filename] = self.params.copy()

        # Prepare success message
        success_msg = f"Results saved for {self.current_filename}: {count} nuclei detected, {len(self.red_dots)} red dots detected"
        if self.analyze_green_dots:
            success_msg += f", {len(self.green_dots)} green dots detected"
            
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
        """Finalize processing and save summary CSV and Excel"""
        if not self.total_results:
            messagebox.showinfo("No Results", "No images were processed.")
            return

        # Save nuclei results to CSV
        results_df = pd.DataFrame(self.total_results)
        csv_path = os.path.join(self.output_dir, "nuclei_counts.csv")
        results_df.to_csv(csv_path, index=False)
        
        # Combine all red dot data into one CSV
        all_red_dots = []
        for filename, dots in self.stored_red_dots.items():
            for dot in dots:
                all_red_dots.append({
                    'Filename': filename,
                    'Color': 'Red',
                    'Dot_ID': dot['id'],
                    'Area_Pixels': dot['area_pixels'],
                    'Area_Microns': dot['area_microns']
                })
        
        # Combine all green dot data into one CSV if applicable
        all_green_dots = []
        if self.analyze_green_dots:
            for filename, dots in self.stored_green_dots.items():
                for dot in dots:
                    all_green_dots.append({
                        'Filename': filename,
                        'Color': 'Green',
                        'Dot_ID': dot['id'],
                        'Area_Pixels': dot['area_pixels'],
                        'Area_Microns': dot['area_microns']
                    })
            
            # Combine red and green dots for overall analysis
            all_dots = all_red_dots + all_green_dots
            if all_dots:
                dots_df = pd.DataFrame(all_dots)
                dots_csv_path = os.path.join(self.output_dir, "all_dots.csv")
                dots_df.to_csv(dots_csv_path, index=False)
        elif all_red_dots:
            # Just save red dots if that's all we have
            red_dots_df = pd.DataFrame(all_red_dots)
            red_dots_csv_path = os.path.join(self.output_dir, "all_red_dots.csv")
            red_dots_df.to_csv(red_dots_csv_path, index=False)
        
        # Create Excel report with requested format
        excel_data = []
        for result in self.total_results:
            filename = result['Filename']
            blue_nuclei_count = result['Total_Nuclei_Count']
            red_dot_count = result['Red_Dot_Count']
            
            # Calculate ratio of red dots to blue nuclei
            red_ratio = 0
            if blue_nuclei_count > 0:
                red_ratio = red_dot_count / blue_nuclei_count
            
            # Calculate average size of red dots in microns
            avg_red_dot_size = 0
            total_red_area = 0
            if filename in self.stored_red_dots and len(self.stored_red_dots[filename]) > 0:
                dots = self.stored_red_dots[filename]
                total_red_area = sum(dot['area_microns'] for dot in dots)
                avg_red_dot_size = total_red_area / len(dots)
            
            # Calculate total red area to blue nuclei ratio
            red_area_to_nuclei_ratio = 0
            if blue_nuclei_count > 0:
                red_area_to_nuclei_ratio = total_red_area / blue_nuclei_count
                
            # Get total green area (always calculated regardless of green dot analysis setting)
            total_green_area = result.get('Total_Green_Area_Microns', 0)
            
            # Calculate total green area to blue nuclei ratio
            green_area_to_nuclei_ratio = 0
            if blue_nuclei_count > 0:
                green_area_to_nuclei_ratio = total_green_area / blue_nuclei_count
            
            # Create the base data for this file
            file_data = {
                'File Name': filename,
                'Total Blue Nuclei': blue_nuclei_count,
                'Total Red Dots': red_dot_count,
                'Total Red Area (μm²)': total_red_area,
                'Red Dots / Blue Nuclei Ratio': red_ratio,
                'Total Red Area / Blue Nuclei (μm²)': red_area_to_nuclei_ratio,
                'Avg Red Dot Size (μm²)': avg_red_dot_size,
                'Total Green Area (μm²)': total_green_area,
                'Total Green Area / Blue Nuclei (μm²)': green_area_to_nuclei_ratio
            }
            
            # Add individual green dot data if applicable
            if self.analyze_green_dots:
                green_dot_count = result.get('Green_Dot_Count', 0)
                
                # Calculate ratio of green dots to blue nuclei
                green_ratio = 0
                if blue_nuclei_count > 0:
                    green_ratio = green_dot_count / blue_nuclei_count
                
                # Calculate average size of green dots in microns
                avg_green_dot_size = 0
                total_analyzed_green_area = 0
                if filename in self.stored_green_dots and len(self.stored_green_dots[filename]) > 0:
                    dots = self.stored_green_dots[filename]
                    total_analyzed_green_area = sum(dot['area_microns'] for dot in dots)
                    avg_green_dot_size = total_analyzed_green_area / len(dots)
                
                # Add green dot columns
                file_data.update({
                    'Total Green Dots': green_dot_count,
                    'Green Dots / Blue Nuclei Ratio': green_ratio,
                    'Avg Green Dot Size (μm²)': avg_green_dot_size
                })
            
            excel_data.append(file_data)
        
        # Create Excel file
        if excel_data:
            excel_df = pd.DataFrame(excel_data)
            excel_path = os.path.join(self.output_dir, "staining_analysis_report.xlsx")
            
            # Format the Excel file with proper column widths
            try:
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    excel_df.to_excel(writer, sheet_name='Analysis Results', index=False)
                    worksheet = writer.sheets['Analysis Results']
                    
                    # Adjust column widths
                    for i, col in enumerate(excel_df.columns):
                        # Set the width based on the length of the column name and the maximum length of data
                        max_length = max(excel_df[col].astype(str).apply(len).max(), len(col)) + 3
                        # Limit width to a reasonable maximum
                        max_length = min(max_length, 50)
                        # Excel measures column width in characters
                        worksheet.column_dimensions[chr(65 + i)].width = max_length
                
                print(f"Excel report saved to {excel_path}")
            except Exception as e:
                print(f"Error creating Excel file: {e}")
                # Fallback to CSV if Excel writing fails
                excel_csv_path = os.path.join(self.output_dir, "staining_analysis_report.csv")
                excel_df.to_csv(excel_csv_path, index=False)
                print(f"Saved as CSV instead: {excel_csv_path}")

        # Calculate summary statistics
        total_count = results_df['Total_Nuclei_Count'].sum()
        auto_count = results_df['Auto_Nuclei_Count'].sum()
        manual_count = results_df['Manual_Nuclei_Count'].sum()
        avg_count = results_df['Total_Nuclei_Count'].mean()
        total_red_dots = results_df['Red_Dot_Count'].sum()
        avg_red_dots = results_df['Red_Dot_Count'].mean()
        avg_total_green_area = results_df['Total_Green_Area_Microns'].mean()
        
        # Green dot statistics if applicable
        total_green_dots = 0
        avg_green_dots = 0
        if self.analyze_green_dots and 'Green_Dot_Count' in results_df.columns:
            total_green_dots = results_df['Green_Dot_Count'].sum()
            avg_green_dots = results_df['Green_Dot_Count'].mean()
        
        # Calculate average dot sizes
        overall_avg_red_dot_size = 0
        overall_avg_green_dot_size = 0
        
        if all_red_dots:
            red_dots_df = pd.DataFrame(all_red_dots)
            overall_avg_red_dot_size = red_dots_df['Area_Microns'].mean()
            
        if self.analyze_green_dots and all_green_dots:
            green_dots_df = pd.DataFrame(all_green_dots)
            overall_avg_green_dot_size = green_dots_df['Area_Microns'].mean()

        # Prepare completion message
        message = (
            f"All images have been processed.\n\n"
            f"Total nuclei detected: {total_count}\n"
            f"- Automatically detected: {auto_count}\n"
            f"- Manually added: {manual_count}\n"
            f"Average nuclei per image: {avg_count:.2f}\n\n"
            f"Total red dots detected: {total_red_dots}\n"
            f"Average red dots per image: {avg_red_dots:.2f}\n"
            f"Average red dot size: {overall_avg_red_dot_size:.2f} μm²\n"
            f"Average total green area per image: {avg_total_green_area:.2f} μm²\n"
        )
        
        if self.analyze_green_dots:
            message += (
                f"\nTotal green dots detected: {total_green_dots}\n"
                f"Average green dots per image: {avg_green_dots:.2f}\n"
                f"Average green dot size: {overall_avg_green_dot_size:.2f} μm²\n"
            )
            
        message += f"\nResults saved to {self.output_dir}\nExcel report: staining_analysis_report.xlsx"
        
        # Show message and schedule application close
        messagebox.showinfo("Processing Complete", message)
        print("Processing completed. Closing application.")
        self.root.after(200, self.close_application)
    
    def close_application(self):
        """Close the application safely"""
        self.root.quit()
        self.root.destroy()
        sys.exit(0)  # Ensure the script terminates


# Run the application
if __name__ == "__main__":
    app = NucleiCounter()