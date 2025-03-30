import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog, simpledialog
import os
import csv
import pandas as pd
from skimage import measure, morphology
from datetime import datetime
from scipy import ndimage
import time

# Default thresholds for color detection - using direct channel thresholding like in analysis_FAST_V2.py
DEFAULT_RED_THRESHOLD = 120
DEFAULT_BLUE_THRESHOLD = 30  # Lowered from 50 to better detect light blue stains
DEFAULT_GREEN_THRESHOLD = 100

# Minimum area (in pixels) to consider a stain valid - to filter out noise
DEFAULT_MIN_RED_SIZE = 2
DEFAULT_MIN_BLUE_SIZE = 500
DEFAULT_MIN_GREEN_SIZE = 20

# Default proximity distance for filtering (in pixels)
DEFAULT_PROXIMITY_DISTANCE = 150

# Conversion factor: 5.4813 pixels/μm
PIXELS_PER_MICRON = 5.4813
MICRONS_PER_PIXEL = 1 / PIXELS_PER_MICRON


def time_function(func):
    """Decorator to time function execution"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        # print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    return wrapper


class StainAnalyzer:
    def __init__(self):
        self.folder_path = None
        self.image_files = []
        self.current_image = None
        self.current_image_path = None
        self.results = {}
        self.satisfied = False

        # Initialize thresholds with default values
        self.thresholds = {
            'red': DEFAULT_RED_THRESHOLD,
            'blue': DEFAULT_BLUE_THRESHOLD,
            'green': DEFAULT_GREEN_THRESHOLD
        }

        # Initialize minimum sizes
        self.min_sizes = {
            'red': DEFAULT_MIN_RED_SIZE,
            'blue': DEFAULT_MIN_BLUE_SIZE,
            'green': DEFAULT_MIN_GREEN_SIZE
        }

        # Proximity distance
        self.proximity_distance = DEFAULT_PROXIMITY_DISTANCE

    def select_folder(self):
        """Select a folder containing TIFF images."""
        root = tk.Tk()
        root.withdraw()
        self.folder_path = filedialog.askdirectory(title="Select folder containing TIFF images")

        if not self.folder_path:
            print("No folder selected. Exiting.")
            return False

        # Get all TIFF files in the folder
        self.image_files = [f for f in os.listdir(self.folder_path)
                            if f.lower().endswith(('.tif', '.tiff'))]

        if not self.image_files:
            print(f"No TIFF images found in {self.folder_path}")
            return False

        print(f"Found {len(self.image_files)} TIFF images in {self.folder_path}")
        return True

    def customize_thresholds(self):
        """Allow the user to customize color detection thresholds."""
        print("\nCurrent color detection thresholds (0-255):")
        print("==========================================")

        print(f"Red threshold: {self.thresholds['red']}")
        print(f"Blue threshold: {self.thresholds['blue']} (lower value helps detect lighter blue stains)")
        print(f"Green threshold: {self.thresholds['green']}")

        print(f"\nMinimum stain sizes (in pixels):")
        print(f"Red: {self.min_sizes['red']}")
        print(f"Blue: {self.min_sizes['blue']}")
        print(f"Green: {self.min_sizes['green']}")

        print(f"\nProximity distance for filtering: {self.proximity_distance} pixels")

        print("\nWould you like to customize these parameters? (Press Enter to use defaults)")
        response = input("Enter 'y' to customize or press Enter to continue: ").lower()

        if response in ['y', 'yes']:
            # Customize thresholds
            for color in ['red', 'blue', 'green']:
                while True:
                    try:
                        value_str = input(
                            f"{color.capitalize()} threshold (0-255, current: {self.thresholds[color]}, press Enter to keep): ")
                        if value_str.strip() == "":
                            break
                        value = int(value_str)
                        if 0 <= value <= 255:
                            self.thresholds[color] = value
                            break
                        else:
                            print("Value must be between 0 and 255.")
                    except ValueError:
                        print("Please enter a valid number.")

            # Customize minimum sizes
            for color in ['red', 'blue', 'green']:
                while True:
                    try:
                        value_str = input(
                            f"Minimum {color} stain size (pixels, current: {self.min_sizes[color]}, press Enter to keep): ")
                        if value_str.strip() == "":
                            break
                        value = int(value_str)
                        if value > 0:
                            self.min_sizes[color] = value
                            break
                        else:
                            print("Minimum size must be positive.")
                    except ValueError:
                        print("Please enter a valid number.")

            # Customize proximity distance
            while True:
                try:
                    value_str = input(
                        f"Proximity distance (pixels, current: {self.proximity_distance}, press Enter to keep): ")
                    if value_str.strip() == "":
                        break
                    value = int(value_str)
                    if value > 0:
                        self.proximity_distance = value
                        break
                    else:
                        print("Proximity distance must be positive.")
                except ValueError:
                    print("Please enter a valid number.")

            print("\nUpdated parameters:")
            print(f"Red threshold: {self.thresholds['red']}")
            print(f"Blue threshold: {self.thresholds['blue']}")
            print(f"Green threshold: {self.thresholds['green']}")
            print(f"Minimum red size: {self.min_sizes['red']} pixels")
            print(f"Minimum blue size: {self.min_sizes['blue']} pixels")
            print(f"Minimum green size: {self.min_sizes['green']} pixels")
            print(f"Proximity distance: {self.proximity_distance} pixels")

        return True

    @time_function
    def load_image(self, image_path):
        """Load an image and return it."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return None
        return image

    @time_function
    def adjust_white_balance(self, image):
        """Adjust white balance using CLAHE like in analysis_FAST_V2.py"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    @time_function
    def detect_stains(self, image, color):
        """
        Detect stains of a specific color in the image using direct channel thresholding.

        Args:
            image: BGR image
            color: 'red', 'blue', or 'green'

        Returns:
            binary mask of the detected stains
        """
        # Define which channel to use based on color
        if color == 'red':
            # Red is channel 2 in BGR
            channel_idx = 2
            threshold = self.thresholds['red']
        elif color == 'blue':
            # Blue is channel 0 in BGR
            channel_idx = 0
            threshold = self.thresholds['blue']

            # For blue stains, we want to ensure it's blue and not cyan (which has high green too)
            # Only consider pixels where blue > green by a good margin
            # This helps distinguish true blue stains from cyan ones
            blue_channel = image[:, :, 0]
            green_channel = image[:, :, 1]
            red_channel = image[:, :, 2]

            # A pixel is blue if:
            # 1. Blue channel > threshold
            # 2. Blue channel > green channel + 10 (to avoid cyan)
            # 3. Blue channel > red channel (to avoid purple)
            return (blue_channel > threshold) & (blue_channel > green_channel + 10) & (blue_channel > red_channel)

        elif color == 'green':
            # Green is channel 1 in BGR
            channel_idx = 1
            threshold = self.thresholds['green']
        else:
            raise ValueError(f"Unsupported color: {color}")

        # For red and green, just use simple thresholding
        if color != 'blue':  # We've already handled blue above
            channel = image[:, :, channel_idx]
            return channel > threshold

    @time_function
    def filter_by_size(self, binary, min_size):
        """Filter binary image to remove objects smaller than min_size"""
        return morphology.remove_small_objects(binary, min_size=min_size)

    @time_function
    def filter_by_proximity(self, target_binary, reference_binary, max_distance):
        """
        Filter target binary to keep only objects within max_distance of reference binary.
        Used to filter red stains based on proximity to blue stains.
        """
        # Calculate distance transform of the inverse of reference binary image
        dist_transform = ndimage.distance_transform_edt(~reference_binary)

        # Create a mask where the distance is less than or equal to max_distance
        proximity_mask = dist_transform <= max_distance

        # Apply the proximity mask to the target binary image
        return target_binary & proximity_mask

    @time_function
    def analyze_stains(self, image, image_name):
        """
        Analyze stains in the image and return results.

        Args:
            image: BGR image
            image_name: name of the image file

        Returns:
            dict with analysis results
        """
        results = {
            'image_name': image_name,
            'red_count': 0,
            'red_areas': [],
            'red_centroids': [],
            'blue_count': 0,
            'blue_centroids': [],
            'green_count': 0,
            'green_areas': [],
            'green_centroids': []
        }

        # Adjust white balance
        balanced = self.adjust_white_balance(image)

        # Detect blue stains first (they're used for proximity filtering)
        blue_mask = self.detect_stains(balanced, 'blue')
        blue_filtered = self.filter_by_size(blue_mask, self.min_sizes['blue'])
        blue_labeled = measure.label(blue_filtered)
        blue_props = measure.regionprops(blue_labeled)
        results['blue_count'] = len(blue_props)
        for prop in blue_props:
            results['blue_centroids'].append(prop.centroid)

        # Detect and filter red stains
        red_mask = self.detect_stains(balanced, 'red')
        red_size_filtered = self.filter_by_size(red_mask, self.min_sizes['red'])
        # Apply proximity filtering to red stains (keep only those near blue stains)
        red_proximity_filtered = self.filter_by_proximity(red_size_filtered, blue_filtered, self.proximity_distance)
        red_labeled = measure.label(red_proximity_filtered)
        red_props = measure.regionprops(red_labeled)
        results['red_count'] = len(red_props)
        for prop in red_props:
            # Convert area from pixels to square microns
            area_microns = prop.area / (PIXELS_PER_MICRON ** 2)
            results['red_areas'].append(area_microns)
            results['red_centroids'].append(prop.centroid)

        # Detect and filter green stains
        green_mask = self.detect_stains(balanced, 'green')
        green_filtered = self.filter_by_size(green_mask, self.min_sizes['green'])
        green_labeled = measure.label(green_filtered)
        green_props = measure.regionprops(green_labeled)
        results['green_count'] = len(green_props)
        for prop in green_props:
            # Convert area from pixels to square microns
            area_microns = prop.area / (PIXELS_PER_MICRON ** 2)
            results['green_areas'].append(area_microns)
            results['green_centroids'].append(prop.centroid)

        # Create visualization images
        original_rgb = cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        vis_images = {
            'original': original_rgb,
            'red': self.create_numbered_overlay(original_rgb, gray_rgb, red_proximity_filtered, (255, 0, 0),
                                                results['red_centroids']),
            'blue': self.create_numbered_overlay(original_rgb, gray_rgb, blue_filtered, (0, 0, 255),
                                                 results['blue_centroids']),
            'green': self.create_numbered_overlay(original_rgb, gray_rgb, green_filtered, (0, 255, 0),
                                                  results['green_centroids'])
        }

        # Print debugging information
        print(f"Image: {image_name}")
        print(f"Initial red stains detected: {np.sum(red_mask)}")
        print(f"Red stains after size filtering: {np.sum(red_size_filtered)}")
        print(f"Red stains after proximity filtering: {np.sum(red_proximity_filtered)}")
        print(f"Final red stain count: {results['red_count']}")
        print(f"Blue stain count: {results['blue_count']}")
        print(f"Green stain count: {results['green_count']}")

        return results, vis_images

    def create_numbered_overlay(self, original, gray_img, mask, color, centroids):
        """
        Create an overlay where the masked areas are in original color, the rest is grayscale,
        and each stain is numbered.

        Args:
            original: RGB image
            gray_img: Grayscale version of the image (in RGB format)
            mask: Binary mask of the areas to highlight
            color: Color to use for highlighting (for visualization)
            centroids: List of (row, col) centroids for each stain

        Returns:
            Image with highlighted areas in color, the rest in grayscale, and numbered stains
        """
        # Create a copy of the grayscale image
        result = gray_img.copy()

        # Replace the masked areas with the original color
        result[mask > 0] = original[mask > 0]

        # Add a colored border around the detected regions for better visibility
        kernel = np.ones((3, 3), np.uint8)
        mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        mask_border = mask_dilated & ~mask

        # Apply the colored border
        result[mask_border > 0] = color

        # Create a copy for adding text (numbers)
        result_with_numbers = result.copy()

        # Add numbers to each stain
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9  # Increased font size
        font_thickness = 2  # Increased thickness

        # Choose contrasting text color based on highlight color
        if color == (255, 0, 0) or color == (0, 0, 255):  # Red or Blue
            font_color = (255, 255, 0)  # Yellow text
        else:  # Green
            font_color = (255, 0, 255)  # Magenta text

        for i, centroid in enumerate(centroids):
            # Convert row, col to x, y coordinates for cv2.putText
            y, x = int(centroid[0]), int(centroid[1])

            # Add a black background for better visibility
            text = str(i + 1)
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            # Position the text with a larger offset from the centroid
            # Offset by 20 pixels to the right and 20 pixels up from the centroid
            text_x = max(0, x + 20)  # Position to the right of the centroid
            text_y = max(text_size[1] + 5, y - 20)  # Position above the centroid

            # Make sure text stays within image boundaries
            h, w = result_with_numbers.shape[:2]
            text_x = min(text_x, w - text_size[0] - 5)
            text_y = min(text_y, h - 5)

            # Draw a filled rectangle as background with larger padding
            cv2.rectangle(result_with_numbers,
                          (text_x - 4, text_y - text_size[1] - 4),
                          (text_x + text_size[0] + 4, text_y + 4),
                          (0, 0, 0),
                          -1)  # Filled rectangle

            # Add a white outline around the black background for better visibility
            cv2.rectangle(result_with_numbers,
                          (text_x - 5, text_y - text_size[1] - 5),
                          (text_x + text_size[0] + 5, text_y + 5),
                          (255, 255, 255),
                          2)  # White outline with increased thickness

            # Add the number
            cv2.putText(result_with_numbers,
                        text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        font_color,
                        font_thickness)

            # Add a line connecting the number to the centroid for clarity
            cv2.line(result_with_numbers,
                     (x, y),
                     (text_x + text_size[0] // 2, text_y - text_size[1] // 2),
                     (255, 255, 255),
                     1)

        return result_with_numbers

    def display_results(self, results, vis_images):
        """Display the analysis results and visualizations."""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig)

        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(vis_images['original'])
        ax1.set_title('Original Image')
        ax1.axis('off')

        # Red stains
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(vis_images['red'])
        ax2.set_title(f'Red Stains (Count: {results["red_count"]})')
        ax2.axis('off')

        # Blue stains
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(vis_images['blue'])
        ax3.set_title(f'Blue Stains (Count: {results["blue_count"]})')
        ax3.axis('off')

        # Green stains
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(vis_images['green'])
        ax4.set_title(f'Green Stains (Count: {results["green_count"]})')
        ax4.axis('off')

        # Results text
        ax5 = fig.add_subplot(gs[1, 1:])
        ax5.axis('off')

        # Format the results text
        text = f"Image: {results['image_name']}\n\n"
        text += f"Red Stains: {results['red_count']}\n"
        if results['red_areas']:
            text += f"Avg Red Area: {np.mean(results['red_areas']):.2f} μm²\n"
            # List individual areas with their numbers
            for i, area in enumerate(results['red_areas']):
                text += f"  #{i + 1}: {area:.2f} μm²\n"

        text += f"\nBlue Stains: {results['blue_count']}\n"

        text += f"\nGreen Stains: {results['green_count']}\n"
        if results['green_areas']:
            text += f"Avg Green Area: {np.mean(results['green_areas']):.2f} μm²\n"
            # List individual areas with their numbers
            for i, area in enumerate(results['green_areas']):
                text += f"  #{i + 1}: {area:.2f} μm²\n"

        ax5.text(0, 1, text, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show(block=False)  # Show the figure but don't block execution

        # After showing the figure, ask for confirmation in the command line
        while True:
            response = input("\nAre you satisfied with the analysis? (y/n): ").lower()
            if response in ['y', 'yes']:
                self.satisfied = True
                break
            elif response in ['n', 'no']:
                self.satisfied = False
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")

        # Close the figure after getting the response
        plt.close(fig)

    def analyze_single_image(self):
        """Analyze a single image from the selected folder."""
        if not self.image_files:
            print("No images to analyze.")
            return False

        # Look for Series003Snapshot1.tif specifically
        target_image = "Series003Snapshot1.tif"
        if target_image in self.image_files:
            image_name = target_image
        else:
            # Fallback to the first image if the target isn't found
            image_name = self.image_files[0]
            print(f"Target image {target_image} not found, using {image_name} instead.")

        image_path = os.path.join(self.folder_path, image_name)

        # Load and analyze the image
        image = self.load_image(image_path)
        if image is None:
            return False

        print(f"Analyzing {image_name}...")
        results, vis_images = self.analyze_stains(image, image_name)

        # Display the results
        self.display_results(results, vis_images)

        # Store the results
        self.results[image_name] = results

        return True

    def analyze_all_images(self):
        """Analyze all images in the selected folder."""
        if not self.image_files:
            print("No images to analyze.")
            return False

        print(f"Analyzing {len(self.image_files)} images...")

        for i, image_name in enumerate(self.image_files):
            image_path = os.path.join(self.folder_path, image_name)

            # Load and analyze the image
            image = self.load_image(image_path)
            if image is None:
                continue

            print(f"Analyzing {i + 1}/{len(self.image_files)}: {image_name}...")
            results, _ = self.analyze_stains(image, image_name)

            # Store the results
            self.results[image_name] = results

        return True

    def save_results_to_csv(self):
        """Save the analysis results to a CSV file."""
        if not self.results:
            print("No results to save.")
            return False

        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.folder_path, f"stain_analysis_{timestamp}.csv")

        # Create a DataFrame to store the results
        df = pd.DataFrame()

        # Add columns for each image
        for image_name, results in self.results.items():
            # Basic counts
            df.loc['Red Count', image_name] = results['red_count']
            df.loc['Blue Count', image_name] = results['blue_count']
            df.loc['Green Count', image_name] = results['green_count']

            # Red areas
            if results['red_areas']:
                df.loc['Avg Red Area (μm²)', image_name] = np.mean(results['red_areas'])
                for i, area in enumerate(results['red_areas']):
                    df.loc[f'Red Area {i + 1} (μm²)', image_name] = area

            # Green areas
            if results['green_areas']:
                df.loc['Avg Green Area (μm²)', image_name] = np.mean(results['green_areas'])
                for i, area in enumerate(results['green_areas']):
                    df.loc[f'Green Area {i + 1} (μm²)', image_name] = area

        # Save to CSV
        df.to_csv(output_path)
        print(f"Results saved to {output_path}")

        # Also save the thresholds used for reference
        threshold_path = os.path.join(self.folder_path, f"thresholds_{timestamp}.txt")
        with open(threshold_path, 'w') as f:
            f.write("Color detection thresholds used for analysis:\n")
            f.write("===========================================\n\n")
            f.write(f"Red threshold: {self.thresholds['red']}\n")
            f.write(f"Blue threshold: {self.thresholds['blue']}\n")
            f.write(f"Green threshold: {self.thresholds['green']}\n\n")
            f.write(f"Minimum red size: {self.min_sizes['red']} pixels\n")
            f.write(f"Minimum blue size: {self.min_sizes['blue']} pixels\n")
            f.write(f"Minimum green size: {self.min_sizes['green']} pixels\n\n")
            f.write(f"Proximity distance: {self.proximity_distance} pixels\n")

        print(f"Parameters saved to {threshold_path}")

        return True

    def run(self):
        """Run the stain analysis."""
        print("Stain Analysis Tool")
        print("==================")
        print("This tool analyzes red, blue, and green stains in TIFF images.")
        print()

        # Select folder
        if not self.select_folder():
            return

        # Allow the user to customize thresholds
        self.customize_thresholds()

        # Analyze a single image first
        if not self.analyze_single_image():
            return

        # Check if the user is satisfied with the results
        if self.satisfied:
            # Analyze all images
            if self.analyze_all_images():
                # Save results to CSV
                self.save_results_to_csv()
        else:
            print("Exiting without analyzing all images.")

        print("Analysis complete!")


def main():
    analyzer = StainAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    main()