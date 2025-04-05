import numpy as np
import cv2
from skimage import io, filters, measure, color, morphology, exposure
from skimage.morphology import skeletonize
from skan import csr
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from matplotlib.colors import ListedColormap
import random
from scipy import ndimage
from skimage.segmentation import watershed
import pandas as pd
from scipy.spatial.distance import cdist

# --- Constants ---
PIXELS_PER_MICRON = 5.7273  # Conversion factor: 1 micron = 5.7273 pixels
MICRONS_PER_PIXEL = 1 / PIXELS_PER_MICRON

# --- File picker ---
Tk().withdraw()
image_path = filedialog.askopenfilename(
    title="Select a mitochondria TIFF image",
    filetypes=[("TIFF files", "*.tif *.tiff")]
)
if not image_path:
    print("No file selected. Exiting.")
    exit()

# --- Load image ---
img = io.imread(image_path)
original_img = img.copy()  # Store original colored image

# --- Convert to grayscale if needed ---
if img.ndim == 3:
    print("Image has multiple channels — converting to grayscale.")
    # Store the blue channel for nuclei detection
    blue_channel = img[:, :, 2]  # BGR format, so blue is channel 2
    img = color.rgb2gray(img)
else:
    print("Image is already grayscale.")
    blue_channel = img
    original_img = np.stack((img,)*3, axis=-1)  # Convert to RGB for visualization

# --- Preprocess following MiNA steps ---
# Normalize to [0,1]
img = img.astype(np.float32)
img = (img - img.min()) / (img.max() - img.min())

# Step B: Unsharp Mask
blurred = filters.gaussian(img, sigma=1)
img_sharpened = img + (img - blurred)
# Normalize after unsharp mask
img_sharpened = (img_sharpened - img_sharpened.min()) / (img_sharpened.max() - img_sharpened.min())

# Step C: CLAHE (Contrast Limited Adaptive Histogram Equalization)
img_clahe = exposure.equalize_adapthist(img_sharpened, clip_limit=0.03)

# Step D: Median Filter
img_median = filters.median(img_clahe)

# Step E: Binary
threshold = filters.threshold_otsu(img_median)
binary = img_median > threshold

# Calculate mitochondrial footprint before skeletonization
footprint_area_pixels = np.sum(binary)
footprint_area_microns = footprint_area_pixels * (MICRONS_PER_PIXEL ** 2)
footprint_percent = 100 * footprint_area_pixels / binary.size

# Step F: Skeletonize
skeleton = skeletonize(binary)

# --- Manual Nuclei Marking ---
def mark_nuclei(image, original_colored):
    """Allow user to draw polygons around nuclei and return their contours"""
    nuclei_contours = []
    current_contour = []
    
    def onclick(event):
        if event.inaxes is not None and event.button == 1:  # Left click
            x, y = int(event.xdata), int(event.ydata)
            current_contour.append([x, y])
            # Draw point
            event.inaxes.plot(x, y, 'y.', markersize=5)
            # Draw line if we have at least 2 points
            if len(current_contour) > 1:
                x1, y1 = current_contour[-2]
                x2, y2 = current_contour[-1]
                event.inaxes.plot([x1, x2], [y1, y2], 'y-', linewidth=1)
            plt.draw()
    
    def onkey(event):
        nonlocal current_contour
        if event.key == 'enter':  # Finish current nucleus
            if len(current_contour) > 2:  # Need at least 3 points for a polygon
                # Close the polygon
                first_point = current_contour[0]
                last_point = current_contour[-1]
                event.inaxes.plot([last_point[0], first_point[0]], 
                                [last_point[1], first_point[1]], 'y-', linewidth=1)
                # Convert to numpy array and add to nuclei_contours
                nuclei_contours.append(np.array(current_contour, dtype=np.int32))
                # Add nucleus number
                center_x = np.mean([p[0] for p in current_contour])
                center_y = np.mean([p[1] for p in current_contour])
                plt.text(center_x, center_y, str(len(nuclei_contours)), 
                        color='yellow', fontsize=10, ha='center', va='center',
                        bbox=dict(facecolor='black', alpha=0.5, pad=1))
                # Reset current_contour for next nucleus
                current_contour = []
                plt.draw()
        elif event.key == 'backspace':  # Remove last point
            if current_contour:
                current_contour.pop()
                ax.cla()  # Clear axis
                ax.imshow(original_colored)  # Redraw original colored image
                # Redraw all completed nuclei
                for i, contour in enumerate(nuclei_contours, 1):
                    ax.plot(contour[:, 0], contour[:, 1], 'y-', linewidth=1)
                    center_x = np.mean(contour[:, 0])
                    center_y = np.mean(contour[:, 1])
                    ax.text(center_x, center_y, str(i), 
                           color='yellow', fontsize=10, ha='center', va='center',
                           bbox=dict(facecolor='black', alpha=0.5, pad=1))
                # Redraw current contour
                if current_contour:
                    points = np.array(current_contour)
                    ax.plot(points[:, 0], points[:, 1], 'y.-', markersize=5, linewidth=1)
                plt.draw()
        elif event.key == 'escape':  # Finish all nuclei marking
            plt.close()
    
    # Create figure for nuclei marking
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(original_colored)
    ax.set_title('Draw nuclei outlines\nClick to add points, Enter to complete current nucleus\nEscape when done with all nuclei')
    
    # Connect events
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    
    # Add instruction text
    plt.figtext(0.5, 0.01,
                'Left click: Add point\n'
                'Enter: Complete current nucleus\n'
                'Backspace: Remove last point\n'
                'Escape: Finish all nuclei',
                ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.show()
    return nuclei_contours

print("Please outline each nucleus in the image.")
print("Click points to draw outline, press Enter to complete each nucleus, Escape when done with all nuclei.")
nuclei_contours = mark_nuclei(img, original_img)
print(f"Marked {len(nuclei_contours)} nuclei")

# Calculate nuclei areas right after getting contours
nuclei_areas = []
for contour in nuclei_contours:
    area_pixels = cv2.contourArea(contour)
    area_microns = area_pixels * (MICRONS_PER_PIXEL ** 2)
    nuclei_areas.append(area_microns)

# Create nuclei mask from contours
nuclei_mask = np.zeros_like(img, dtype=np.uint8)
cv2.drawContours(nuclei_mask, nuclei_contours, -1, 255, -1)  # -1 fills the contours

# --- Analyze skeleton ---
# Find junction pixels using hit-miss transform
from skimage.morphology import thin
from scipy.ndimage import convolve

def find_junctions(skel):
    """Returns coordinates of junction pixels (pixels with more than 2 neighbors)"""
    kernel = np.array([[1, 1, 1],
                      [1, 10, 1],
                      [1, 1, 1]])
    neighbor_count = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
    return (neighbor_count > 12) & skel  # More than 2 neighbors (excluding center)

def count_branches(skel, junctions):
    """Count number of branches in a skeleton component"""
    # Find endpoints
    kernel = np.array([[1, 1, 1],
                      [1, 10, 1],
                      [1, 1, 1]])
    neighbor_count = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
    endpoints = (neighbor_count == 11) & skel  # Only one neighbor
    
    # Number of branches is:
    # (number of endpoints + 2 * number of junction points) / 2
    n_endpoints = np.sum(endpoints)
    n_junctions = np.sum(junctions)
    
    return (n_endpoints + 2 * n_junctions) // 2

# Identify junctions
junction_pixels = find_junctions(skeleton)

# Label connected components in skeleton
labeled_skeleton = measure.label(skeleton)
props = measure.regionprops(labeled_skeleton)

# Initialize lists for analysis
individuals = []  # Components with no junctions (0 or 1 branch)
networks = []    # Components with junctions (multiple branches)
network_branches = []  # Number of branches per network

# Analyze each component
for prop in props:
    # Get the region mask and corresponding junction pixels
    region = prop.image
    region_junctions = junction_pixels[prop.bbox[0]:prop.bbox[2], 
                                    prop.bbox[1]:prop.bbox[3]] & region
    
    # Count branches in this component
    n_branches = count_branches(region, region_junctions)
    
    # Classify based on junctions
    if np.any(region_junctions):
        networks.append(prop)
        network_branches.append(n_branches)
    else:
        individuals.append(prop)

# Calculate statistics
n_individuals = len(individuals)
n_networks = len(networks)

# Network size statistics (number of branches per network)
if network_branches:
    mean_network_size = np.mean(network_branches)
    median_network_size = np.median(network_branches)
    std_network_size = np.std(network_branches)
else:
    mean_network_size = median_network_size = std_network_size = 0

# Branch length statistics (for visualization)
skeleton_data = csr.Skeleton(skeleton)
branch_lengths_pixels = skeleton_data.path_lengths()
branch_lengths_microns = branch_lengths_pixels * MICRONS_PER_PIXEL

# --- Generate Excel Report ---
def get_component_centroid(component):
    """Calculate centroid of a region property component"""
    return np.mean(component.coords, axis=0)

def get_network_stats(network_indices, network_branches, branch_lengths_microns):
    """Calculate network statistics for a subset of networks"""
    if len(network_indices) == 0:
        return 0, 0, 0  # No networks
    
    relevant_branches = [network_branches[i] for i in network_indices]
    relevant_lengths = branch_lengths_microns[network_indices]
    
    mean_size = np.mean(relevant_branches) if relevant_branches else 0
    mean_length = np.mean(relevant_lengths) if len(relevant_lengths) > 0 else 0
    
    return len(network_indices), mean_size, mean_length

# Calculate centroids for all components
network_centroids = np.array([get_component_centroid(net) for net in networks]) if networks else np.empty((0, 2))
individual_centroids = np.array([get_component_centroid(ind) for ind in individuals]) if individuals else np.empty((0, 2))

# Prepare data for Excel
excel_data = []

# Process marked nuclei
for nucleus_id, (contour, area) in enumerate(zip(nuclei_contours, nuclei_areas), 1):
    # Calculate nucleus centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:
        nucleus_cx = int(M["m10"] / M["m00"])
        nucleus_cy = int(M["m01"] / M["m00"])
        nucleus_centroid = np.array([nucleus_cy, nucleus_cx])  # Note: y,x format to match region props
        
        # Find associated networks
        if len(networks) > 0:
            network_distances = cdist([nucleus_centroid], network_centroids)
            associated_networks = np.where(network_distances[0] < 100)[0]  # Within 100 pixels
        else:
            associated_networks = np.array([], dtype=int)
            
        # Find associated individuals
        if len(individuals) > 0:
            individual_distances = cdist([nucleus_centroid], individual_centroids)
            associated_individuals = np.where(individual_distances[0] < 100)[0]  # Within 100 pixels
        else:
            associated_individuals = np.array([], dtype=int)
        
        # Calculate footprint for associated components
        footprint = 0
        for idx in associated_networks:
            footprint += networks[idx].area
        for idx in associated_individuals:
            footprint += individuals[idx].area
        footprint_microns = footprint * (MICRONS_PER_PIXEL ** 2)
        
        # Get network statistics
        n_networks, mean_network_size, mean_branch_length = get_network_stats(
            associated_networks, network_branches, branch_lengths_microns)
        
        # Add row to Excel data
        excel_data.append({
            'Nucleus ID': nucleus_id,
            'Nucleus Area (μm²)': area,
            'Mitochondrial Footprint (μm²)': footprint_microns,
            'Individuals': len(associated_individuals),
            'Networks': n_networks,
            'Mean Network Size': mean_network_size,
            'Mean Branch Length (μm)': mean_branch_length
        })

# Process unassociated networks
if len(networks) > 0:
    # Find networks that weren't associated with any nucleus
    all_associated = set()
    for row in excel_data:
        nucleus_centroid = np.array([nucleus_cy, nucleus_cx])
        associated_networks = np.where(cdist([nucleus_centroid], network_centroids) < 100)[0]
        all_associated.update(associated_networks)
    
    unassociated = set(range(len(networks))) - all_associated
    
    # Add unassociated networks to Excel data
    for net_idx in unassociated:
        network = networks[net_idx]
        footprint = network.area * (MICRONS_PER_PIXEL ** 2)
        _, mean_size, mean_length = get_network_stats(
            np.array([net_idx]), network_branches, branch_lengths_microns)
        
        excel_data.append({
            'Nucleus ID': None,
            'Nucleus Area (μm²)': None,
            'Mitochondrial Footprint (μm²)': footprint,
            'Individuals': 0,
            'Networks': 1,
            'Mean Network Size': mean_size,
            'Mean Branch Length (μm)': mean_length
        })

# Create and save Excel file
df = pd.DataFrame(excel_data)
excel_path = image_path.rsplit('.', 1)[0] + '_analysis.xlsx'
df.to_excel(excel_path, index=False)
print(f"\nAnalysis saved to: {excel_path}")

# --- Visualization ---
fig = plt.figure(figsize=(20, 10))  # Made figure wider to accommodate new subplot

# Original with overlay
ax1 = plt.subplot(241)
ax1.imshow(original_img)
ax1.imshow(binary, cmap='magma', alpha=0.2)
ax1.set_title("Original + Binary Mask")
ax1.axis('off')

# Binary
ax2 = plt.subplot(242)
ax2.imshow(binary, cmap='gray')
ax2.set_title("Binary")
ax2.axis('off')

# Skeleton with junctions
ax3 = plt.subplot(243)
ax3.imshow(skeleton, cmap='gray')
ax3.imshow(junction_pixels, cmap='Reds', alpha=0.7)
ax3.set_title("Skeleton + Junctions")
ax3.axis('off')

# Simple Skeleton (Black on White)
ax4 = plt.subplot(244)
ax4.imshow(~skeleton, cmap='binary')  # Invert skeleton to make it black on white
ax4.set_title("Skeleton")
ax4.axis('off')

# Networks and Individuals
ax5 = plt.subplot(245)
network_mask = np.zeros_like(skeleton)
individual_mask = np.zeros_like(skeleton)

for network in networks:
    coords = network.coords
    network_mask[coords[:, 0], coords[:, 1]] = 1
for individual in individuals:
    coords = individual.coords
    individual_mask[coords[:, 0], coords[:, 1]] = 1

ax5.imshow(original_img, cmap='gray', alpha=0.5)
ax5.imshow(network_mask, cmap='Reds', alpha=0.5)
ax5.imshow(individual_mask, cmap='Blues', alpha=0.5)
ax5.set_title(f"Networks ({n_networks}) and\nIndividuals ({n_individuals})")
ax5.axis('off')

# Network size distribution
ax6 = plt.subplot(246)
if network_branches:
    counts, bins, _ = ax6.hist(network_branches, bins='auto',
                              color='lightcoral', alpha=0.7, edgecolor='black')
    ax6.axvline(mean_network_size, color='black', linestyle='--',
                label=f'Mean: {mean_network_size:.1f}')
    ax6.set_title("Network Size Distribution\n(Branches per Network)")
    ax6.set_xlabel("Number of Branches")
    ax6.set_ylabel("Frequency")
    ax6.legend()
else:
    ax6.text(0.5, 0.5, "No networks found", ha='center', va='center')
    ax6.set_title("Network Size Distribution")
ax6.axis('on')

# Branch length distribution
ax7 = plt.subplot(247)
if len(branch_lengths_microns) > 0:
    counts, bins, _ = ax7.hist(branch_lengths_microns, bins='auto',
                              color='lightgreen', alpha=0.7, edgecolor='black')
    ax7.set_title("Branch Length Distribution")
    ax7.set_xlabel("Length (μm)")
    ax7.set_ylabel("Frequency")
else:
    ax7.text(0.5, 0.5, "No branches found", ha='center', va='center')
    ax7.set_title("Branch Length Distribution")
ax7.axis('on')

# Nuclei Areas
ax8 = plt.subplot(248)
nuclei_centroids = []

# Get areas from contours
contours, _ = cv2.findContours(nuclei_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area_pixels = cv2.contourArea(contour)
    area_microns = area_pixels * (MICRONS_PER_PIXEL ** 2)
    nuclei_areas.append(area_microns)
    
    # Calculate centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        nuclei_centroids.append((cx, cy))

# Create nuclei visualization
nuclei_img = cv2.cvtColor(img_median.astype(np.float32), cv2.COLOR_GRAY2BGR)
for i, (contour, area, centroid) in enumerate(zip(contours, nuclei_areas, nuclei_centroids)):
    cv2.drawContours(nuclei_img, [contour], -1, (0, 255, 255), 2)  # Yellow contour
    cx, cy = centroid
    cv2.putText(nuclei_img, f"{area:.1f}μm²", (cx-20, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

ax8.imshow(nuclei_img)
ax8.set_title("Nuclei Areas")
ax8.axis('off')

plt.tight_layout()
plt.show()

# Print results
print(f"\n📄 File: {image_path}")
print("\nMitochondrial Metrics:")
print(f"1. Mitochondrial footprint:")
print(f"   - Area (pixels²): {footprint_area_pixels:.2f}")
print(f"   - Area (μm²): {footprint_area_microns:.2f}")
print(f"   - Percent of image: {footprint_percent:.2f}%")
print(f"2. Number of individuals (puncta and rods): {n_individuals}")
print(f"3. Number of networks: {n_networks}")
print("\nNetwork Size Statistics:")
print(f"4. Mean branches per network: {mean_network_size:.2f}")
print(f"5. Median branches per network: {median_network_size:.2f}")
print(f"6. Network size standard deviation: {std_network_size:.2f}")
print("\nNuclei Statistics:")
if nuclei_areas:
    print(f"7. Number of nuclei: {len(nuclei_areas)}")
    print(f"8. Average nucleus area: {np.mean(nuclei_areas):.2f} μm²")
    print(f"9. Total nuclei area: {np.sum(nuclei_areas):.2f} μm²")