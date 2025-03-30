# Proteoastat analysis
import cv2
import numpy as np
from skimage import measure, morphology
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import csv
import time
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage

MICRON_CONVERSION = 5.7273  # 1 micron = 5.7273 pixels


# Function to print styled text
def print_stylish_text():
    print("//--###################################--//")
    print("||                                     ||")
    print("||    WELCOME TO GORDON & LANDMAN      ||")
    print("||          STAIN ANALYSIS             ||")
    print("||                                     ||")
    print("//--###################################--//")
    print()


def print_funny_closing():
    print()
    print("****************************************")
    print("||  Analysis Complete!                  ||")
    print("||  Go grab a coffee, you've earned it! ||")
    print("||                                      ||")
    print("||          ( (                         ||")
    print("||           ) )                        ||")
    print("||        .______.                      ||")
    print("||        |      |]                     ||")
    print("||        \      /                      ||")
    print("||         `-----'                      ||")
    print("****************************************")


def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select folder containing TIFF images")
    return folder_path


def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        # print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    return wrapper


@time_function
def adjust_white_balance(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


@time_function
def detect_stains(image, color, threshold):
    color_dict = {'blue': 0, 'red': 2}  # BGR order for cv2
    if color not in color_dict:
        raise ValueError("Color must be 'red' or 'blue'")

    channel = color_dict[color]
    return image[:, :, channel] > threshold


@time_function
def filter_by_size(binary, min_size):
    return morphology.remove_small_objects(binary, min_size=min_size)


@time_function
def filter_by_proximity(red_binary, blue_binary, max_distance):
    # Calculate distance transform of the inverse of blue binary image
    dist_transform = ndimage.distance_transform_edt(~blue_binary)

    # Create a mask where the distance is less than or equal to max_distance
    proximity_mask = dist_transform <= max_distance

    # Apply the proximity mask to the red binary image
    return red_binary & proximity_mask


@time_function
def analyze_image(image_path, min_blue_size, proximity_distance, blue_threshold, red_threshold, min_red_size,
                  output_folder):
    original = cv2.imread(image_path)
    if original is None:
        raise IOError(f"Failed to read image: {image_path}")

    balanced = adjust_white_balance(original)

    blue_mask = detect_stains(balanced, 'blue', blue_threshold)
    blue_filtered = filter_by_size(blue_mask, min_blue_size)
    blue_labeled = measure.label(blue_filtered)
    blue_count = np.max(blue_labeled)

    red_mask = detect_stains(balanced, 'red', red_threshold)
    red_size_filtered = filter_by_size(red_mask, min_red_size)
    red_proximity_filtered = filter_by_proximity(red_size_filtered, blue_filtered, proximity_distance)
    red_labeled = measure.label(red_proximity_filtered)
    red_props = measure.regionprops(red_labeled)
    red_count = len(red_props)

    # Debugging information
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Initial red stains detected: {np.sum(red_mask)}")
    print(f"Red stains after size filtering: {np.sum(red_size_filtered)}")
    print(f"Red stains after proximity filtering: {np.sum(red_proximity_filtered)}")
    print(f"Final red stain count: {red_count}")

    csv_data = [
        ['type', 'value'],
        ['count blue', blue_count],
        ['count red', red_count]
    ]
    for i, prop in enumerate(red_props, 1):
        size_microns = prop.area / MICRON_CONVERSION
        csv_data.append([f'red_{i:03d}', f'{size_microns:.2f}'])

    plot_results(balanced, blue_filtered, red_proximity_filtered, output_folder, os.path.basename(image_path))
    save_csv(csv_data, output_folder, os.path.basename(image_path))

    return blue_count, red_count


@time_function
def plot_results(balanced, blue_filtered, red_filtered, output_folder, image_name):
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes[0, 0].imshow(cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original")

    axes[0, 1].imshow(cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Blue Stains")
    blue_contours, _ = cv2.findContours(blue_filtered.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in blue_contours:
        axes[0, 1].plot(contour[:, 0, 0], contour[:, 0, 1], 'y', linewidth=2)

    axes[1, 0].imshow(cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Red Stains")
    red_contours, _ = cv2.findContours(red_filtered.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in red_contours:
        axes[1, 0].plot(contour[:, 0, 0], contour[:, 0, 1], 'g', linewidth=2)

    axes[1, 1].imshow(red_filtered, cmap='gray')
    axes[1, 1].set_title(f"Red Stain Mask (Count: {len(red_contours)})")

    plt.tight_layout()
    output_image_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_analysis.png")
    plt.savefig(output_image_path)
    plt.close(fig)


@time_function
def save_csv(csv_data, output_folder, image_name):
    csv_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)


def get_user_input(prompt, default):
    user_input = input(f"{prompt} (default: {default}): ")
    return int(user_input) if user_input else default


def process_file(args):
    filename, folder_path, min_blue_size, proximity_distance, blue_threshold, red_threshold, min_red_size, output_folder = args
    if filename.lower().endswith(('.tif', '.tiff')):
        image_path = os.path.join(folder_path, filename)
        print(f"Processing {filename}...")
        try:
            blue_count, red_count = analyze_image(image_path, min_blue_size, proximity_distance,
                                                  blue_threshold, red_threshold, min_red_size, output_folder)
            print(f"Blue stains: {blue_count}")
            print(f"Red stains: {red_count}")
            print("---")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")


def create_excel_summary(output_folder):
    try:
        import pandas as pd
        summary_data = []

        for filename in os.listdir(output_folder):
            if filename.endswith('.csv'):
                csv_path = os.path.join(output_folder, filename)
                df = pd.read_csv(csv_path)

                img_name = os.path.splitext(filename)[0]
                count_blue = df[df['type'] == 'count blue']['value'].values[0]
                count_red = df[df['type'] == 'count red']['value'].values[0]

                summary_data.append({
                    'Image Name': img_name,
                    'Count Blue': count_blue,
                    'Count Red': count_red
                })

        summary_df = pd.DataFrame(summary_data)
        excel_path = os.path.join(output_folder, 'stain_count_summary.xlsx')
        summary_df.to_excel(excel_path, index=False)
        print(f"Excel summary created: {excel_path}")
    except ImportError:
        print("Warning: Unable to create Excel summary. Please install required libraries:")
        print("pip install pandas openpyxl")
        print("Falling back to CSV summary...")
        create_csv_summary(output_folder, summary_data)
    except Exception as e:
        print(f"Error creating Excel summary: {str(e)}")
        print("Falling back to CSV summary...")
        create_csv_summary(output_folder, summary_data)


def create_csv_summary(output_folder, summary_data):
    csv_path = os.path.join(output_folder, 'stain_count_summary.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Image Name', 'Count Blue', 'Count Red'])
        writer.writeheader()
        writer.writerows(summary_data)
    print(f"CSV summary created: {csv_path}")


def main():
    # Print stylish welcome message
    print_stylish_text()

    # User input defaults
    DEFAULT_MIN_BLUE_SIZE = 500
    DEFAULT_PROXIMITY_DISTANCE = 150
    DEFAULT_BLUE_THRESHOLD = 50
    DEFAULT_RED_THRESHOLD = 120
    DEFAULT_MIN_RED_SIZE = 2

    min_blue_size = get_user_input("Enter the minimum blue stain size (in pixels)", DEFAULT_MIN_BLUE_SIZE)
    proximity_distance = get_user_input("Enter the proximity distance for red stains (in pixels)",
                                        DEFAULT_PROXIMITY_DISTANCE)
    blue_threshold = get_user_input("Enter the blue color threshold (0-255)", DEFAULT_BLUE_THRESHOLD)
    red_threshold = get_user_input("Enter the red color threshold (0-255)", DEFAULT_RED_THRESHOLD)
    min_red_size = get_user_input("Enter the minimum red stain size (in pixels)", DEFAULT_MIN_RED_SIZE)

    folder_path = select_folder()
    if not folder_path:
        print("No folder selected. Exiting.")
        return

    output_folder = os.path.join(folder_path, "analysis_results")
    os.makedirs(output_folder, exist_ok=True)

    filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))]

    start_time = time.time()

    with ProcessPoolExecutor() as executor:
        args_list = [
            (filename, folder_path, min_blue_size, proximity_distance, blue_threshold, red_threshold, min_red_size,
             output_folder) for
            filename in filenames]
        executor.map(process_file, args_list)

    end_time = time.time()

    # Create Excel summary
    create_excel_summary(output_folder)

    # Print funny closing message
    print_funny_closing()

    print(f"Results saved in {output_folder}")
    print(f"Total execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()