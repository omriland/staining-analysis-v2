import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog, Scale, Button, Label, Frame, Toplevel, messagebox
from tkinter.messagebox import askyesno
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage import feature, measure, morphology, segmentation
from scipy.ndimage import distance_transform_edt


class NucleiCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nuclei Counter")
        self.root.geometry("800x600")

        # Default parameters
        self.blue_threshold = 150  # For blue channel thresholding
        self.min_size = 100  # Minimum nucleus size in pixels
        self.max_size = 1000  # Maximum nucleus size in pixels
        self.separation_factor = 5  # For separating connected nuclei

        self.images_folder = ""
        self.output_folder = ""
        self.image_files = []
        self.preview_images = []
        self.current_preview_index = 0

        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        header_label = Label(main_frame, text="Nuclei Counter", font=("Arial", 16, "bold"))
        header_label.pack(pady=10)

        # Folder selection
        folder_frame = Frame(main_frame)
        folder_frame.pack(fill=tk.X, pady=5)

        Label(folder_frame, text="Select folder with TIFF images:").pack(side=tk.LEFT)
        Button(folder_frame, text="Browse", command=self.select_folder).pack(side=tk.RIGHT)

        # Selected folder display
        self.folder_label = Label(main_frame, text="No folder selected", fg="gray")
        self.folder_label.pack(fill=tk.X, pady=5)

        # Parameter adjustment frame
        param_frame = Frame(main_frame)
        param_frame.pack(fill=tk.X, pady=10)

        param_header = Label(param_frame, text="Adjust Parameters", font=("Arial", 12, "bold"))
        param_header.pack(pady=5)

        # Blue threshold slider
        thresh_frame = Frame(param_frame)
        thresh_frame.pack(fill=tk.X, pady=5)
        Label(thresh_frame, text="Blue Channel Threshold:").pack(side=tk.LEFT)
        self.thresh_slider = Scale(thresh_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                   length=300, variable=tk.IntVar(value=self.blue_threshold))
        self.thresh_slider.pack(side=tk.RIGHT)

        # Min size slider
        min_size_frame = Frame(param_frame)
        min_size_frame.pack(fill=tk.X, pady=5)
        Label(min_size_frame, text="Minimum Nucleus Size (pixels):").pack(side=tk.LEFT)
        self.min_size_slider = Scale(min_size_frame, from_=10, to=500, orient=tk.HORIZONTAL,
                                     length=300, variable=tk.IntVar(value=self.min_size))
        self.min_size_slider.pack(side=tk.RIGHT)

        # Max size slider
        max_size_frame = Frame(param_frame)
        max_size_frame.pack(fill=tk.X, pady=5)
        Label(max_size_frame, text="Maximum Nucleus Size (pixels):").pack(side=tk.LEFT)
        self.max_size_slider = Scale(max_size_frame, from_=500, to=5000, orient=tk.HORIZONTAL,
                                     length=300, variable=tk.IntVar(value=self.max_size))
        self.max_size_slider.pack(side=tk.RIGHT)

        # Separation factor slider
        sep_frame = Frame(param_frame)
        sep_frame.pack(fill=tk.X, pady=5)
        Label(sep_frame, text="Separation Factor (lower for more separation):").pack(side=tk.LEFT)
        self.sep_slider = Scale(sep_frame, from_=1, to=20, orient=tk.HORIZONTAL,
                                length=300, variable=tk.IntVar(value=self.separation_factor))
        self.sep_slider.pack(side=tk.RIGHT)

        # Action buttons
        button_frame = Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        Button(button_frame, text="Show Preview", command=self.show_preview, width=15).pack(side=tk.LEFT, padx=5)
        Button(button_frame, text="Process All Images", command=self.process_all_images, width=15).pack(side=tk.RIGHT,
                                                                                                        padx=5)

    def select_folder(self):
        self.images_folder = filedialog.askdirectory(title="Select folder with TIFF images")
        if self.images_folder:
            self.folder_label.config(text=f"Selected: {self.images_folder}", fg="black")

            # Find all TIFF files
            self.image_files = [f for f in os.listdir(self.images_folder)
                                if f.lower().endswith(('.tif', '.tiff'))]

            if not self.image_files:
                messagebox.showinfo("No Images", "No TIFF images found in the selected folder.")
            else:
                # Create output folder
                self.output_folder = os.path.join(self.images_folder, "nuclei_results")
                os.makedirs(self.output_folder, exist_ok=True)

                # Select two images for preview
                self.preview_images = self.image_files[:2] if len(self.image_files) >= 2 else self.image_files
                self.current_preview_index = 0

                messagebox.showinfo("Files Found",
                                    f"Found {len(self.image_files)} TIFF images. Click 'Show Preview' to test current settings.")

    def show_preview(self):
        if not self.preview_images:
            messagebox.showinfo("No Preview", "Please select a folder with TIFF images first.")
            return

        # Get current parameter values
        self.blue_threshold = self.thresh_slider.get()
        self.min_size = self.min_size_slider.get()
        self.max_size = self.max_size_slider.get()
        self.separation_factor = self.sep_slider.get()

        # Create preview window
        preview_window = Toplevel(self.root)
        preview_window.title("Preview - Nuclei Detection")
        preview_window.geometry("1000x800")

        # Process current preview image
        current_image = self.preview_images[self.current_preview_index]
        img_path = os.path.join(self.images_folder, current_image)

        try:
            # Process image and get results
            result_dict = self.process_image(img_path)

            # Create figure for displaying images
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

            # Display original image
            ax1.imshow(result_dict['original'])
            ax1.set_title("Original Image")
            ax1.axis('off')

            # Display blue channel
            ax2.imshow(result_dict['blue_enhanced'])
            ax2.set_title("Enhanced Blue Channel")
            ax2.axis('off')

            # Display binary mask
            ax3.imshow(result_dict['binary'], cmap='gray')
            ax3.set_title(f"Thresholded (value: {self.blue_threshold})")
            ax3.axis('off')

            # Display processed image with detections
            ax4.imshow(result_dict['processed'])
            ax4.set_title(f"Detected Nuclei: {result_dict['count']}")
            ax4.axis('off')

            # Add canvas to window
            canvas = FigureCanvasTkAgg(fig, master=preview_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Feedback text
            feedback_text = """Adjustment Guide:
- If some nuclei are not detected (missing yellow outlines): Decrease the Blue Threshold
- If too many non-nuclei regions are detected: Increase the Blue Threshold
- If connected nuclei are not separated: Decrease the Separation Factor
- If single nuclei are split into multiple: Increase the Separation Factor
- If small artifacts are detected: Increase Minimum Nucleus Size
- If large nuclei are not detected: Increase Maximum Nucleus Size"""

            Label(preview_window, text=feedback_text, justify=tk.LEFT, font=("Arial", 10)).pack(pady=10)

            # Navigation and action buttons
            button_frame = Frame(preview_window)
            button_frame.pack(fill=tk.X, pady=10)

            # Show image name and navigation controls
            image_label = Label(button_frame,
                                text=f"Image: {current_image} ({self.current_preview_index + 1}/{len(self.preview_images)})")
            image_label.pack(side=tk.TOP, pady=5)

            nav_frame = Frame(button_frame)
            nav_frame.pack(side=tk.TOP, pady=5)

            Button(nav_frame, text="Previous Image",
                   command=lambda: self.navigate_preview(-1, preview_window)).pack(side=tk.LEFT, padx=5)

            Button(nav_frame, text="Next Image",
                   command=lambda: self.navigate_preview(1, preview_window)).pack(side=tk.LEFT, padx=5)

            # Action buttons
            action_frame = Frame(button_frame)
            action_frame.pack(side=tk.BOTTOM, pady=10)

            Button(action_frame, text="Accept Settings", width=15,
                   command=lambda: self.accept_settings(preview_window)).pack(side=tk.LEFT, padx=10)

            Button(action_frame, text="Adjust Parameters", width=15,
                   command=preview_window.destroy).pack(side=tk.RIGHT, padx=10)

        except Exception as e:
            preview_window.destroy()
            messagebox.showerror("Error", f"Error processing image: {str(e)}")

    def navigate_preview(self, direction, window):
        # Change preview image index
        self.current_preview_index = (self.current_preview_index + direction) % len(self.preview_images)
        window.destroy()
        self.show_preview()

    def accept_settings(self, window):
        # Ask for confirmation
        response = askyesno("Confirm",
                            "Are you satisfied with the current settings?\nDo you want to process all images?")
        if response:
            window.destroy()
            self.process_all_images()

    def process_image(self, image_path):
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Add debug print to help diagnose issues
        print(f"Image shape: {img.shape}, dtype: {img.dtype}")
        print(f"Image min/max values: {np.min(img)}/{np.max(img)}")

        # Convert BGR to RGB for display
        original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # For fluorescence images, we want the blue channel
        # In BGR format, the channels are:
        # B: 0, G: 1, R: 2
        # Let's extract the blue channel
        blue_channel = img[:, :, 0].copy()

        # Enhance contrast in the blue channel
        blue_channel_enhanced = cv2.equalizeHist(blue_channel)

        # Create a debug image to see what's happening with the blue channel
        blue_enhanced_img = original_rgb.copy()
        blue_enhanced_img[:, :,
        2] = blue_channel_enhanced  # Enhance the red channel in the debug image for better visualization

        # Threshold the blue channel
        _, binary = cv2.threshold(blue_channel_enhanced, self.blue_threshold, 255, cv2.THRESH_BINARY)

        # Clean up binary image
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # Remove small objects
        labeled = measure.label(opening > 0)
        filtered = morphology.remove_small_objects(labeled, min_size=self.min_size)
        mask = filtered > 0

        # Apply distance transform to separate connected nuclei
        distance = distance_transform_edt(mask)

        # Create peaks for watershed
        # The separation factor is used as min_distance - lower value will find more peaks
        coords = feature.peak_local_max(
            distance,
            min_distance=self.separation_factor,
            labels=filtered
        )

        # Create markers for watershed
        markers = np.zeros_like(mask, dtype=np.uint8)
        if len(coords) > 0:  # Check if any peaks were found
            markers[tuple(coords.T)] = 1
        markers = measure.label(markers)

        # Apply watershed segmentation
        segmented = segmentation.watershed(-distance, markers, mask=mask)

        # Filter objects by size
        props = measure.regionprops(segmented)
        valid_regions = []

        for prop in props:
            if self.min_size <= prop.area <= self.max_size:
                valid_regions.append(prop.label)

        # Create mask for valid nuclei
        nucleus_mask = np.zeros_like(segmented, dtype=bool)
        for label in valid_regions:
            nucleus_mask[segmented == label] = True

        # Create output image with overlay
        processed = original_rgb.copy()

        # Draw yellow outlines around detected nuclei
        for label in valid_regions:
            # Create mask for the current region
            region_mask = (segmented == label).astype(np.uint8)
            # Find contours
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw contour
            cv2.drawContours(processed, contours, -1, (255, 255, 0), 2)

        # Return all results in a dictionary
        return {
            'original': original_rgb,
            'blue_enhanced': blue_enhanced_img,
            'binary': binary,
            'processed': processed,
            'count': len(valid_regions)
        }

    def process_all_images(self):
        if not self.image_files:
            messagebox.showinfo("No Images", "Please select a folder with TIFF images first.")
            return

        # Get current parameter values
        self.blue_threshold = self.thresh_slider.get()
        self.min_size = self.min_size_slider.get()
        self.max_size = self.max_size_slider.get()
        self.separation_factor = self.sep_slider.get()

        # Create results dataframe
        results = []

        # Process each image
        progress_window = Toplevel(self.root)
        progress_window.title("Processing Images")
        progress_window.geometry("600x100")

        progress_label = Label(progress_window, text="Processing images...")
        progress_label.pack(pady=20)

        progress_window.update()

        successful = 0
        for i, image_file in enumerate(self.image_files):
            # Update progress
            progress_label.config(text=f"Processing image {i + 1}/{len(self.image_files)}: {image_file}")
            progress_window.update()

            # Process image
            img_path = os.path.join(self.images_folder, image_file)
            try:
                result_dict = self.process_image(img_path)
                processed = result_dict['processed']
                count = result_dict['count']

                # Save processed image
                output_path = os.path.join(self.output_folder, f"marked_{image_file}")
                cv2.imwrite(output_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))

                # Add result to dataframe
                results.append({"Image": image_file, "Nuclei Count": count})
                successful += 1
            except Exception as e:
                messagebox.showerror("Error", f"Error processing {image_file}: {str(e)}")

        # Save results to CSV
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.output_folder, "nuclei_counts.csv")
        df.to_csv(csv_path, index=False)

        progress_window.destroy()

        # Show completion message
        message = f"Processing complete!\n\n"
        message += f"Successfully processed {successful}/{len(self.image_files)} images.\n"
        message += f"Results saved to {self.output_folder}\n\n"
        message += f"CSV file: {os.path.basename(csv_path)}"

        messagebox.showinfo("Processing Complete", message)


if __name__ == "__main__":
    root = tk.Tk()
    app = NucleiCounterApp(root)
    root.mainloop()