#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Required imports
import os
import sys
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

matplotlib.use('TkAgg')


class NucleiCounter:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window

        # Default parameters - updated based on user feedback
        self.params = {
            'blue_threshold': 65,  # Threshold for blue channel
            'min_size': 153,  # Minimum nucleus size in pixels
            'max_size': 10000,  # Maximum nucleus size in pixels
            'dilation_size': 5,  # Size of dilation kernel to connect fragments
            'closing_size': 4,  # Size of closing kernel
            'distance_threshold': 15  # Distance threshold for connecting fragments
        }

        # UI elements
        self.preview_fig = None
        self.preview_axs = None
        self.binary_img = None
        self.result_img = None
        self.param_window = None
        self.navigation_window = None
        self.sliders = {}
        self.selected_image = None

        # Data
        self.current_image = None
        self.current_filename = None
        self.output_dir = None
        self.total_results = []
        self.process_all = False
        self.current_file_index = 0
        self.input_dir = None
        self.tiff_files = None

        # User-added nuclei storage
        self.manual_nuclei = []
        self.stored_manual_nuclei = {}

    def select_input_folder(self):
        """Let user select a folder containing TIFF images"""
        input_dir = filedialog.askdirectory(title="Select folder with TIFF images")
        if not input_dir:
            print("No folder selected. Exiting.")
            sys.exit()

        # Create output directory
        self.output_dir = os.path.join(input_dir, "nuclei_results")
        os.makedirs(self.output_dir, exist_ok=True)

        # Get list of TIFF files
        tiff_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))]
        if not tiff_files:
            print(f"No TIFF files found in {input_dir}. Exiting.")
            sys.exit()

        return input_dir, tiff_files

    def adjust_parameters(self):
        """GUI for adjusting parameters with live preview updates"""
        param_window = tk.Toplevel(self.root)
        param_window.title("Adjust Parameters")
        param_window.geometry("400x400")

        # Create a frame for parameters
        frame = ttk.Frame(param_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # Add sliders for each parameter
        row = 0
        sliders = {}

        # Parameter ranges and step sizes
        param_configs = {
            'blue_threshold': (0, 255, 1),
            'min_size': (10, 500, 10),
            'max_size': (500, 10000, 100),
            'dilation_size': (0, 20, 1),
            'closing_size': (0, 20, 1),
            'distance_threshold': (1, 30, 1)
        }

        # First, load image and initialize preview figure
        if self.current_image is None:
            print("No image loaded for preview")
            return

        # Initialize the preview figure
        if self.preview_fig is None:
            self.preview_fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            self.preview_fig.canvas.manager.set_window_title("Live Preview")
            self.preview_axs = axs.ravel()

            # Display original image with correct colors (static)
            if len(self.current_image[0].shape) == 3:
                # Handle the image based on how it was loaded
                self.preview_axs[0].imshow(self.current_image[0])
            else:
                self.preview_axs[0].imshow(self.current_image[0], cmap='gray')
            self.preview_axs[0].set_title("Original Image")

            # Display blue channel (static)
            self.preview_axs[1].imshow(self.current_image[1], cmap='Blues')
            self.preview_axs[1].set_title("Blue Channel (DAPI)")

            # Initialize placeholders for dynamic images
            self.binary_img = self.preview_axs[2].imshow(np.zeros_like(self.current_image[1]), cmap='gray')
            self.preview_axs[2].set_title("Binary (Thresholded) Image")

            self.result_img = self.preview_axs[3].imshow(self.current_image[0])
            self.preview_axs[3].set_title("Detection Result")

            # Add click event for adding missed nuclei
            self.preview_fig.canvas.mpl_connect('button_press_event', self._on_click)

            # Add text about manual marking feature
            self.preview_fig.text(0.5, 0.01,
                                  "Click on missed nuclei in the Detection Result to mark them manually",
                                  ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

            # Calculate current count for the title
            _, _, _, count = self.process_image(self.current_image[0], self.current_image[1])
            self.preview_fig.suptitle(
                f"Preview: {self.current_filename} - {count} nuclei detected\n"
                f"(Auto: {count - len(self.manual_nuclei)}, Manual: {len(self.manual_nuclei)})",
                fontsize=14
            )

            plt.tight_layout()
            self.preview_fig.subplots_adjust(top=0.85, bottom=0.08)
            plt.ion()  # Turn on interactive mode
            plt.show()

        # Create sliders with live update functionality
        for param, value in self.params.items():
            label = ttk.Label(frame, text=param.replace('_', ' ').title())
            label.grid(row=row, column=0, sticky=tk.W, pady=5)

            min_val, max_val, step = param_configs[param]

            var = tk.IntVar(value=value)
            slider = ttk.Scale(
                frame, from_=min_val, to=max_val,
                variable=var, orient=tk.HORIZONTAL,
                length=200, command=lambda v, p=param, var=var: self._update_param_live(p, int(float(var.get())))
            )
            slider.grid(row=row, column=1, padx=5, pady=5)

            value_label = ttk.Label(frame, text=str(value))
            value_label.grid(row=row, column=2, padx=5)

            sliders[param] = (var, value_label)
            row += 1

            # Set initial value and trace changes
            var.trace_add("write", lambda *args, p=param, v=var, l=value_label:
            l.config(text=str(int(float(v.get())))))

        # Add a button to clear manual nuclei
        clear_manual_btn = ttk.Button(frame, text="Clear Manual Markers", command=self._clear_manual_nuclei)
        clear_manual_btn.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        # Update preview initially
        self._update_preview()

        # Keep reference to widgets
        self.param_window = param_window
        self.sliders = sliders

        return param_window

    def _on_click(self, event):
        """Handle click events to add missed nuclei"""
        # Only accept clicks on the detection result panel (index 3)
        if event.inaxes == self.preview_axs[3]:
            x, y = int(event.xdata), int(event.ydata)

            # Check if within image bounds
            if 0 <= x < self.current_image[0].shape[1] and 0 <= y < self.current_image[0].shape[0]:
                # Add to manual nuclei list
                self.manual_nuclei.append((x, y))
                print(f"Manual nucleus added at ({x}, {y})")

                # Update preview
                self._update_preview()

                # Update navigation status to show updated count
                if hasattr(self, 'count_label'):
                    _, _, _, count = self.process_image(self.current_image[0], self.current_image[1])
                    count_text = f"Nuclei count: {count} (Auto: {count - len(self.manual_nuclei)}, Manual: {len(self.manual_nuclei)})"
                    self.count_label.config(text=count_text)

    def _clear_manual_nuclei(self):
        """Clear all manually added nuclei markers"""
        self.manual_nuclei = []
        self._update_preview()

        # Update navigation status to show updated count
        if hasattr(self, 'count_label'):
            _, _, _, count = self.process_image(self.current_image[0], self.current_image[1])
            count_text = f"Nuclei count: {count} (Auto: {count}, Manual: 0)"
            self.count_label.config(text=count_text)

    def _update_param_live(self, param, value):
        """Update parameter value and refresh preview in real-time"""
        self.params[param] = value
        self._update_preview()

    def _update_preview(self):
        """Update the preview with current parameters"""
        if self.current_image is None:
            return

        # Process the image with current parameters
        binary, nuclei_mask, result_img, count = self.process_image(
            self.current_image[0], self.current_image[1])

        # Update the preview figure
        if self.preview_fig:
            # Update binary image
            self.binary_img.set_data(binary)

            # Update result image with correct color display
            if len(result_img.shape) == 3:
                # If using OpenCV to create result image, convert BGR to RGB for display
                if np.array_equal(result_img[:, :, 0], self.current_image[0][:, :, 2]) and \
                        np.array_equal(result_img[:, :, 2], self.current_image[0][:, :, 0]):
                    # This indicates BGR order needs conversion
                    self.result_img.set_data(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                else:
                    # Image is already in correct color order
                    self.result_img.set_data(result_img)
            else:
                self.result_img.set_data(result_img)

            # Update title with count information
            self.preview_fig.suptitle(
                f"Preview: {self.current_filename} - {count} nuclei detected\n"
                f"(Auto: {count - len(self.manual_nuclei)}, Manual: {len(self.manual_nuclei)})",
                fontsize=14
            )

            # Refresh the canvas
            self.preview_fig.canvas.draw_idle()
            self.preview_fig.canvas.flush_events()

            # Update navigation status if available
            if hasattr(self, 'count_label'):
                count_text = f"Nuclei count: {count} (Auto: {count - len(self.manual_nuclei)}, Manual: {len(self.manual_nuclei)})"
                self.count_label.config(text=count_text)

    def show_preview(self):
        """This method is kept for backward compatibility but is now handled automatically
        by the live preview functionality"""
        self._update_preview()

    def _on_process_all(self):
        """Close parameter window and signal to process all files"""
        # Store the manually marked nuclei for each image if any
        if self.current_filename and self.manual_nuclei:
            self.stored_manual_nuclei[self.current_filename] = self.manual_nuclei.copy()
        else:
            self.stored_manual_nuclei = {}

        self.param_window.destroy()
        if hasattr(self, 'navigation_window') and self.navigation_window:
            self.navigation_window.destroy()
        if self.preview_fig:
            plt.close(self.preview_fig)
        self.process_all = True

    def load_image(self, image_path):
        """Load an image and extract the blue channel"""
        try:
            # Read the image using scikit-image (loads in RGB format)
            image = io.imread(image_path)

            # Handle grayscale images
            if len(image.shape) == 2:
                return image, image

            # For color images, extract blue channel
            if len(image.shape) == 3:
                if image.shape[2] == 3:  # RGB
                    # In RGB order, blue is channel 2 (index 2)
                    blue_channel = image[:, :, 2]
                    return image, blue_channel
                elif image.shape[2] == 4:  # RGBA
                    blue_channel = image[:, :, 2]
                    return image, blue_channel

            print(f"Unexpected image format: {image.shape}")
            return None, None

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None

    def process_image(self, image, blue_channel):
        """Process image to detect nuclei"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(blue_channel, (5, 5), 0)

        # Apply threshold to get binary image
        _, binary = cv2.threshold(blurred, self.params['blue_threshold'], 255, cv2.THRESH_BINARY)

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

        # Draw contours and number them
        for i, contour in enumerate(contours):
            cv2.drawContours(result_img, [contour], -1, (0, 255, 255), 2)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(result_img, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Add manually marked nuclei
        for i, (x, y) in enumerate(self.manual_nuclei):
            # Draw a circle for each manually added nucleus
            cv2.circle(result_img, (x, y), 15, (0, 255, 255), 2)
            # Label with number continuing from automatic detections
            cv2.putText(result_img, str(len(contours) + i + 1), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return binary, nuclei_mask, result_img, len(contours) + len(self.manual_nuclei)

    def process_files(self, input_dir, tiff_files):
        """Process files one by one with manual review"""
        self.process_all = False
        self.stored_manual_nuclei = {}
        self.input_dir = input_dir
        self.tiff_files = tiff_files
        self.current_file_index = 0

        # Load the first image for preview
        self._load_current_image()

        # Show parameter adjustment window with navigation
        self.setup_navigation_window(input_dir, tiff_files)
        param_window = self.adjust_parameters()
        self.root.wait_window(param_window)

        # Close preview figure if still open
        if self.preview_fig:
            plt.close(self.preview_fig)
            self.preview_fig = None

        # Process all remaining files if requested
        if self.process_all:
            self._process_remaining_files()

        # Save results to CSV
        if self.total_results:
            results_df = pd.DataFrame(self.total_results)
            csv_path = os.path.join(self.output_dir, "nuclei_counts.csv")
            results_df.to_csv(csv_path, index=False)

            print(f"Processing complete. Results saved to {self.output_dir}")
            print(f"CSV results saved to {csv_path}")

            # Show summary statistics
            total_count = results_df['Total_Nuclei_Count'].sum()
            auto_count = results_df['Auto_Nuclei_Count'].sum()
            manual_count = results_df['Manual_Nuclei_Count'].sum()
            avg_count = results_df['Total_Nuclei_Count'].mean()
            print(f"Total nuclei detected: {total_count}")
            print(f"  - Automatically detected: {auto_count}")
            print(f"  - Manually added: {manual_count}")
            print(f"Average nuclei per image: {avg_count:.2f}")

    def _load_current_image(self):
        """Load the current image based on index"""
        if self.current_file_index < len(self.tiff_files):
            filename = self.tiff_files[self.current_file_index]
            image_path = os.path.join(self.input_dir, filename)

            # Store current manual nuclei before changing images
            if self.current_filename and self.manual_nuclei:
                self.stored_manual_nuclei[self.current_filename] = self.manual_nuclei.copy()

            # Load new image
            self.current_filename = filename
            self.current_image = self.load_image(image_path)

            # Retrieve any stored manual nuclei for this image
            if filename in self.stored_manual_nuclei:
                self.manual_nuclei = self.stored_manual_nuclei[filename].copy()
            else:
                self.manual_nuclei = []

            return True
        return False

    def _save_current_image(self):
        """Save results for the current image and return True if successful"""
        if self.current_image is None or self.current_filename is None:
            return False

        # Process image with current parameters
        binary, nuclei_mask, result_img, count = self.process_image(
            self.current_image[0], self.current_image[1])

        # Save result image
        output_path = os.path.join(self.output_dir, f"result_{self.current_filename}")
        io.imsave(output_path, result_img)

        # Store results
        result_entry = {
            'Filename': self.current_filename,
            'Auto_Nuclei_Count': count - len(self.manual_nuclei),
            'Manual_Nuclei_Count': len(self.manual_nuclei),
            'Total_Nuclei_Count': count
        }

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

        print(f"Saved results for {self.current_filename}: {count} nuclei detected")
        return True

    def _move_to_next_image(self):
        """Save current image and move to the next one"""
        # Save current image
        if not self._save_current_image():
            return False

        # Move to next image
        self.current_file_index += 1
        if self._load_current_image():
            # Update preview
            self._update_image_display()
            self._update_preview()
            return True
        else:
            # No more images
            self._finalize_processing()
            return False

    def _move_to_prev_image(self):
        """Save current image and move to the previous one"""
        # Save current image
        if not self._save_current_image():
            return False

        # Move to previous image
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self._load_current_image()
            # Update preview
            self._update_image_display()
            self._update_preview()
            return True
        return False

    def _update_image_display(self):
        """Update the image display when changing images"""
        if self.preview_fig and self.current_image:
            # Update static parts (original and blue channel)
            self.preview_axs[0].clear()
            if len(self.current_image[0].shape) == 3:
                # Display with correct colors
                self.preview_axs[0].imshow(self.current_image[0])
            else:
                self.preview_axs[0].imshow(self.current_image[0], cmap='gray')
            self.preview_axs[0].set_title("Original Image")

            self.preview_axs[1].clear()
            self.preview_axs[1].imshow(self.current_image[1], cmap='Blues')
            self.preview_axs[1].set_title("Blue Channel (DAPI)")

            # Update dynamic parts
            self._update_preview()

            # Update navigation status
            self._update_navigation_status()

    def _finalize_processing(self):
        """Handle completion of all images"""
        print("All images have been processed.")
        # If there's a UI, update it to show completion
        if hasattr(self, 'navigation_window') and self.navigation_window:
            messagebox.showinfo("Processing Complete",
                                f"All {len(self.tiff_files)} images have been processed.")
            self.navigation_window.destroy()

        # Close parameter window if it exists
        if hasattr(self, 'param_window') and self.param_window:
            self.param_window.destroy()

        # Close preview
        if self.preview_fig:
            plt.close(self.preview_fig)
            self.preview_fig = None

    def _process_remaining_files(self):
        """Process all remaining files without further user interaction"""
        start_index = self.current_file_index
        for i in range(start_index, len(self.tiff_files)):
            self.current_file_index = i
            if self._load_current_image():
                self._save_current_image()
                print(f"Auto-processed {self.current_filename}")

        print(f"Processed {len(self.tiff_files) - start_index} remaining files.")

    def setup_navigation_window(self, input_dir, tiff_files):
        """Create a window with navigation buttons for stepping through images"""
        navigation_window = tk.Toplevel(self.root)
        navigation_window.title("Image Navigation")
        navigation_window.geometry("500x150")

        frame = ttk.Frame(navigation_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # Current image indicator
        status_text = f"Image 1 of {len(tiff_files)}: {self.current_filename}"
        self.status_label = ttk.Label(frame, text=status_text)
        self.status_label.grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=10)

        # Nuclei count indicator
        _, _, _, count = self.process_image(self.current_image[0], self.current_image[1])
        self.count_label = ttk.Label(
            frame,
            text=f"Nuclei count: {count} (Auto: {count - len(self.manual_nuclei)}, Manual: {len(self.manual_nuclei)})",
            font=("Arial", 12, "bold")
        )
        self.count_label.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=5)

        # Navigation buttons
        prev_btn = ttk.Button(frame, text="← Previous", command=self._move_to_prev_image)
        prev_btn.grid(row=2, column=0, padx=5, pady=10)

        save_btn = ttk.Button(frame, text="Save Current", command=self._save_current_image)
        save_btn.grid(row=2, column=1, padx=5, pady=10)

        next_btn = ttk.Button(frame, text="Save & Next →", command=self._move_to_next_image)
        next_btn.grid(row=2, column=2, padx=5, pady=10)

        process_all_btn = ttk.Button(frame, text="Process All Remaining", command=self._on_process_remaining)
        process_all_btn.grid(row=2, column=3, padx=5, pady=10)

        self.navigation_window = navigation_window

    def _update_navigation_status(self):
        """Update the navigation status and count labels"""
        if hasattr(self, 'status_label'):
            status_text = f"Image {self.current_file_index + 1} of {len(self.tiff_files)}: {self.current_filename}"
            self.status_label.config(text=status_text)

        if hasattr(self, 'count_label'):
            _, _, _, count = self.process_image(self.current_image[0], self.current_image[1])
            count_text = f"Nuclei count: {count} (Auto: {count - len(self.manual_nuclei)}, Manual: {len(self.manual_nuclei)})"
            self.count_label.config(text=count_text)

    def _on_process_remaining(self):
        """Process all remaining images without further user interaction"""
        response = messagebox.askyesno("Process Remaining",
                                       f"Process the remaining {len(self.tiff_files) - self.current_file_index} images with current settings?")
        if response:
            self._process_remaining_files()
            self._finalize_processing()

    def run(self):
        """Main function to run the nuclei counter"""
        print("=== DAPI-Stained Nuclei Counter ===")

        # Select input folder with TIFF files
        input_dir, tiff_files = self.select_input_folder()
        print(f"Selected folder: {input_dir}")
        print(f"Found {len(tiff_files)} TIFF files")

        # Process files
        self.process_files(input_dir, tiff_files)

        print("Program finished.")


if __name__ == "__main__":
    counter = NucleiCounter()
    counter.run()