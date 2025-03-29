import os
import sys
import numpy as np
import pandas as pd
import cv2
from skimage import io, morphology, measure, segmentation, filters, feature
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
import tkinter as tk
from tkinter import filedialog, simpledialog
from tkinter import ttk
import matplotlib

matplotlib.use('TkAgg')


class NucleiCounter:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window

        # Default parameters
        self.params = {
            'blue_threshold': 50,  # Threshold for blue channel
            'min_size': 100,  # Minimum nucleus size in pixels
            'max_size': 3000,  # Maximum nucleus size in pixels
            'dilation_size': 5,  # Size of dilation kernel to connect fragments
            'closing_size': 7,  # Size of closing kernel
            'distance_threshold': 7  # Distance threshold for connecting fragments
        }

        # UI elements
        self.preview_fig = None
        self.preview_axs = None
        self.binary_img = None
        self.result_img = None
        self.param_window = None
        self.selector_window = None
        self.sliders = {}
        self.selected_image = None

        # Data
        self.current_image = None
        self.current_filename = None
        self.output_dir = None
        self.total_results = []
        self.process_all = False

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
        param_window.geometry("400x350")

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
                # If using cv2 (BGR), convert to RGB, if using io.imread (RGB), keep as is
                # We assume io.imread was used which loads in RGB format by default
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

            plt.tight_layout()
            self.preview_fig.subplots_adjust(top=0.9)
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

        # Update preview initially
        self._update_preview()

        # Add process all button
        process_button = ttk.Button(frame, text="Process All Files", command=self._on_process_all)
        process_button.grid(row=row, column=0, columnspan=3, pady=15)

        # Keep reference to widgets
        self.param_window = param_window
        self.sliders = sliders

        return param_window

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

            # Update title with count
            self.preview_fig.suptitle(f"Preview: {self.current_filename} - {count} nuclei detected")

            # Refresh the canvas
            self.preview_fig.canvas.draw_idle()
            self.preview_fig.canvas.flush_events()

    def _on_process_all(self):
        """Close parameter window and signal to process all files"""
        self.param_window.destroy()
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

        return binary, nuclei_mask, result_img, len(contours)

    def show_preview(self):
        """This method is kept for backward compatibility but is now handled automatically
        by the live preview functionality"""
        self._update_preview()

    def process_files(self, input_dir, tiff_files):
        """Process all files or show preview"""
        self.process_all = False

        # Load the first image for preview
        first_image_path = os.path.join(input_dir, tiff_files[0])
        self.current_filename = tiff_files[0]
        self.current_image = self.load_image(first_image_path)

        # Create a dropdown to select images for preview
        if len(tiff_files) > 1:
            self.setup_image_selector(input_dir, tiff_files)

        # Show parameter adjustment window with live preview
        param_window = self.adjust_parameters()
        self.root.wait_window(param_window)

        # Close preview figure if still open
        if self.preview_fig:
            plt.close(self.preview_fig)
            self.preview_fig = None

        # If process_all is True, process all files
        if self.process_all:
            print(f"Processing {len(tiff_files)} files...")

            for i, filename in enumerate(tiff_files):
                image_path = os.path.join(input_dir, filename)
                print(f"Processing {i + 1}/{len(tiff_files)}: {filename}")

                # Load and process image
                image, blue_channel = self.load_image(image_path)
                if image is None:
                    continue

                binary, nuclei_mask, result_img, count = self.process_image(image, blue_channel)

                # Save result image
                output_path = os.path.join(self.output_dir, f"result_{filename}")
                io.imsave(output_path, result_img)

                # Store results
                self.total_results.append({
                    'Filename': filename,
                    'Nuclei_Count': count
                })

            # Save results to CSV
            results_df = pd.DataFrame(self.total_results)
            csv_path = os.path.join(self.output_dir, "nuclei_counts.csv")
            results_df.to_csv(csv_path, index=False)

            print(f"Processing complete. Results saved to {self.output_dir}")
            print(f"CSV results saved to {csv_path}")

            # Show summary statistics
            total_count = results_df['Nuclei_Count'].sum()
            avg_count = results_df['Nuclei_Count'].mean()
            print(f"Total nuclei detected: {total_count}")
            print(f"Average nuclei per image: {avg_count:.2f}")

    def setup_image_selector(self, input_dir, tiff_files):
        """Create a dropdown to select different images for preview"""
        selector_window = tk.Toplevel(self.root)
        selector_window.title("Image Selector")
        selector_window.geometry("400x100")

        frame = ttk.Frame(selector_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Select image for preview:").grid(row=0, column=0, sticky=tk.W, pady=5)

        # Create string variable for dropdown
        self.selected_image = tk.StringVar(value=tiff_files[0])

        # Create dropdown
        dropdown = ttk.Combobox(frame, textvariable=self.selected_image)
        dropdown['values'] = tiff_files
        dropdown.grid(row=0, column=1, padx=5, pady=5)

        # Load button
        load_btn = ttk.Button(
            frame,
            text="Load Selected Image",
            command=lambda: self._load_selected_image(input_dir)
        )
        load_btn.grid(row=1, column=0, columnspan=2, pady=10)

        self.selector_window = selector_window

    def _load_selected_image(self, input_dir):
        """Load the selected image and update preview"""
        filename = self.selected_image.get()
        image_path = os.path.join(input_dir, filename)

        # Load new image
        self.current_filename = filename
        self.current_image = self.load_image(image_path)

        # Update preview if it exists
        if self.preview_fig:
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
            self.preview_axs[1].set_title("Blue Channel")

            # Update dynamic parts
            self._update_preview()

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