# s129 area / prpf area (%)
# TH percentage
# annexin area per cell?
import os
import numpy as np
from skimage import io, img_as_float, color
import matplotlib.pyplot as plt
import csv
import tkinter as tk
from tkinter import filedialog

def load_image(image_path):
    """Load the image and return the red and green channels."""
    image = io.imread(image_path)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Image must be RGB")
    
    red_channel = img_as_float(image[:, :, 0])
    green_channel = img_as_float(image[:, :, 1])
    return image, red_channel, green_channel

def apply_threshold(channel, thresh):
    """Apply the given threshold to a channel."""
    return channel > (thresh / 255.0)  # Convert threshold to 0-1 range

def quantify_signals(red_channel, green_channel, red_thresh, green_thresh):
    """Quantify the red and green signals."""
    total_pixels = red_channel.size
    
    red_mask = apply_threshold(red_channel, red_thresh)
    green_mask = apply_threshold(green_channel, green_thresh)
    
    red_area = np.sum(red_mask)
    green_area = np.sum(green_mask)
    
    red_percentage = (red_area / total_pixels) * 100
    green_percentage = (green_area / total_pixels) * 100
    
    return {
        'Red Area': red_area,
        'Green Area': green_area,
        'Red Percentage': red_percentage,
        'Green Percentage': green_percentage
    }, red_mask, green_mask

def create_visualization(image, red_mask, green_mask, output_path=None):
    """Create and save a visualization of the original image and thresholded channels."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Red channel visualization
    red_viz = color.gray2rgb(color.rgb2gray(image))
    red_viz[red_mask, 0] = 1.0  # Set red channel to maximum for selected pixels
    red_viz[red_mask, 1:] = 0.0  # Set green and blue channels to minimum for selected pixels
    ax2.imshow(red_viz)
    ax2.set_title('Red Channel')
    ax2.axis('off')
    
    # Green channel visualization
    green_viz = color.gray2rgb(color.rgb2gray(image))
    green_viz[green_mask, 1] = 1.0  # Set green channel to maximum for selected pixels
    green_viz[green_mask, 0] = 0.0  # Set red channel to minimum for selected pixels
    green_viz[green_mask, 2] = 0.0  # Set blue channel to minimum for selected pixels
    ax3.imshow(green_viz)
    ax3.set_title('Green Channel')
    ax3.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def process_image(image_path, output_folder, red_thresh, green_thresh):
    """Process a single image and return its results."""
    try:
        image, red_channel, green_channel = load_image(image_path)
        results, red_mask, green_mask = quantify_signals(red_channel, green_channel, red_thresh, green_thresh)
        results['Image Name'] = os.path.basename(image_path)
        
        # Create visualization
        vis_path = os.path.join(output_folder, f"{os.path.splitext(results['Image Name'])[0]}_visualization.png")
        create_visualization(image, red_mask, green_mask, vis_path)
        
        return results
    except Exception as e:
        print(f"An error occurred while processing {image_path}: {e}")
        return None

def process_folder(folder_path, output_folder, red_thresh, green_thresh):
    """Process all .tif images in the given folder and combine results."""
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.tif', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            result = process_image(image_path, output_folder, red_thresh, green_thresh)
            if result:
                results.append(result)
    
    return results

def select_folder(title):
    """Open a dialog to select a folder."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title=title)
    return folder_path

def get_threshold_input(channel, default):
    """Get threshold input from user with validation."""
    while True:
        try:
            value = input(f"Enter {channel} channel threshold (0-255, default {default}): ")
            if value == "":
                return default
            value = int(value)
            if 0 <= value <= 255:
                return value
            else:
                print("Please enter a value between 0 and 255.")
        except ValueError:
            print("Please enter a valid integer.")

def save_to_csv(results, output_path):
    """Save results to a CSV file."""
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['Image Name', 'Red Area (pixels)', 'Red Percentage (%)', 'Green Area (pixels)', 'Green Percentage (%)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'Image Name': result['Image Name'],
                'Red Area (pixels)': result['Red Area'],
                'Red Percentage (%)': result['Red Percentage'],
                'Green Area (pixels)': result['Green Area'],
                'Green Percentage (%)': result['Green Percentage']
            })
    print(f"Results saved to {output_path}")

def preview_thresholds(image_path):
    """Preview thresholds on a sample image."""
    image, red_channel, green_channel = load_image(image_path)
    
    while True:
        red_thresh = get_threshold_input("red", 50)  # More conservative default
        green_thresh = get_threshold_input("green", 90)  # More conservative default
        
        _, red_mask, green_mask = quantify_signals(red_channel, green_channel, red_thresh, green_thresh)
        create_visualization(image, red_mask, green_mask)
        
        if input("Are you satisfied with these thresholds? (y/n): ").lower() == 'y':
            return red_thresh, green_thresh

def main():
    print("Please select the input folder containing .tif images.")
    folder_path = select_folder("Select Input Folder")
    if not folder_path:
        print("No folder selected. Exiting.")
        return

    print("Please select the output folder for results.")
    output_folder = select_folder("Select Output Folder")
    if not folder_path:
        print("No output folder selected. Exiting.")
        return

    os.makedirs(output_folder, exist_ok=True)
    
    # Preview thresholds on a sample image
    sample_image = next(f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff')))
    sample_path = os.path.join(folder_path, sample_image)
    red_thresh, green_thresh = preview_thresholds(sample_path)
    
    print(f"Using red threshold: {red_thresh}, green threshold: {green_thresh}")
    
    results = process_folder(folder_path, output_folder, red_thresh, green_thresh)
    if results:
        output_path = os.path.join(output_folder, 'image_analysis_results.csv')
        save_to_csv(results, output_path)
    else:
        print("No valid results to save.")

if __name__ == "__main__":
    main()