# Nomber of s129 per cell
# s129 area per cell (micrometer2 analysis)
# annexin per cell
# annexin area per cell?

import cv2
import numpy as np
from skimage import io, measure
import os
import csv
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

MICRON_CONVERSION = 5.7273  # 1 micron = 5.7273 pixels

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select folder containing TIFF images")
    return folder_path

def get_user_input(prompt, default):
    user_input = input(f"{prompt} (default: {default}): ")
    return int(user_input) if user_input else default

def detect_stains(image, color, threshold):
    if color == 'blue':
        color_channel = image[:,:,2]
    elif color == 'green':
        color_channel = image[:,:,1]
    else:
        raise ValueError("Color must be 'blue' or 'green'")
    
    mask = color_channel > threshold
    return mask

def filter_by_size(binary, min_size):
    labeled = measure.label(binary)
    props = measure.regionprops(labeled)
    filtered_mask = np.zeros_like(binary)
    for prop in props:
        if prop.area >= min_size:
            filtered_mask[labeled == prop.label] = 1
    return filtered_mask

def analyze_image(image_path, min_blue_size, blue_threshold, green_threshold):
    # Read the image
    image = io.imread(image_path)
    
    # Detect and filter blue stains
    blue_mask = detect_stains(image, 'blue', blue_threshold)
    blue_filtered = filter_by_size(blue_mask, min_blue_size)
    blue_count = measure.label(blue_filtered).max()
    
    # Detect green dots
    green_mask = detect_stains(image, 'green', green_threshold)
    green_labeled = measure.label(green_mask)
    green_props = measure.regionprops(green_labeled)
    green_count = len(green_props)
    
    # Prepare CSV data
    csv_data = [
        ['Type', 'Value'],
        ['blue count', blue_count],
        ['green count', green_count]
    ]
    for i, prop in enumerate(green_props, 1):
        size_microns = prop.area / MICRON_CONVERSION
        csv_data.append([f'Green_{i:03d}', f'{size_microns:.2f}'])
    
    return blue_filtered, green_mask, csv_data

def generate_preview(image_path, blue_mask, green_mask, output_path):
    image = io.imread(image_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original Image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Blue Stains
    axes[1].imshow(image)
    axes[1].imshow(blue_mask, alpha=0.5, cmap='Blues')
    axes[1].set_title("Blue Stains")
    axes[1].axis('off')
    
    # Green Dots
    axes[2].imshow(image)
    
    # Find contours of green dots
    green_contours = measure.find_contours(green_mask, 0.5)
    
    # Plot green dot contours
    for contour in green_contours:
        axes[2].plot(contour[:, 1], contour[:, 0], linewidth=2, color='lime')
    
    axes[2].set_title("Green Dots")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def process_folder(folder_path, min_blue_size, blue_threshold, green_threshold):
    # Create 'Results' subfolder
    results_folder = os.path.join(folder_path, 'Results')
    os.makedirs(results_folder, exist_ok=True)
    
    tiff_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))]
    
    for filename in tiff_files:
        image_path = os.path.join(folder_path, filename)
        print(f"Processing {filename}...")
        
        blue_mask, green_mask, csv_data = analyze_image(image_path, min_blue_size, blue_threshold, green_threshold)
        
        # Save CSV
        base_name = os.path.splitext(filename)[0]
        csv_path = os.path.join(results_folder, f"{base_name}_analysis.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)
        
        # Generate preview
        preview_path = os.path.join(results_folder, f"{base_name}_preview.png")
        generate_preview(image_path, blue_mask, green_mask, preview_path)
        
        print(f"Analysis complete for {filename}")
        print("---")
    
    print("All images processed.")
    print(f"Results saved in: {results_folder}")

def main():
    # Default values
    DEFAULT_MIN_BLUE_SIZE = 1000
    DEFAULT_BLUE_THRESHOLD = 50
    DEFAULT_GREEN_THRESHOLD = 150

    # Get user input for parameters
    min_blue_size = get_user_input("Enter the minimum blue stain size (in pixels)", DEFAULT_MIN_BLUE_SIZE)
    blue_threshold = get_user_input("Enter the blue color threshold (0-255)", DEFAULT_BLUE_THRESHOLD)
    green_threshold = get_user_input("Enter the green color threshold (0-255)", DEFAULT_GREEN_THRESHOLD)

    # Select folder containing TIFF images
    folder_path = select_folder()
    if not folder_path:
        print("No folder selected. Exiting.")
        return
    
    # Process the folder
    process_folder(folder_path, min_blue_size, blue_threshold, green_threshold)

if __name__ == "__main__":
    main()