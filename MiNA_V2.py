# MiNA_V2.py
# Last updated: 2025-09-20
# This script allows for a one Tiff file to be loaded and analyzed.
# The user is prompted to outline each nucleus in the image.
# The script then calculates the area and centroid of each nucleus.
# The script then calculates the number of branches in each nucleus.
# The script then calculates the number of junctions in each nucleus.
# The script then calculates the number of networks in each nucleus.
# The script then calculates the number of individuals in each nucleus.
# The script then calculates the number of mitochondria in each nucleus.
# The script then outputs the results to an Excel file.

import time
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
skeleton_pixels = np.sum(skeleton)
print(f"Image: {binary.shape[0]}x{binary.shape[1]} | Skeleton: {skeleton_pixels:,} pixels")

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
print("Analyzing skeleton and branches...")
_t_start = _t0 = time.time()

# Calculate nuclei areas right after getting contours
nuclei_areas = []
nuclei_centroids = []
for contour in nuclei_contours:
    # Calculate area
    area_pixels = cv2.contourArea(contour)
    area_microns = area_pixels * (MICRONS_PER_PIXEL ** 2)
    nuclei_areas.append(area_microns)

    # Calculate centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        nuclei_centroids.append((cx, cy))
    else:
        nuclei_centroids.append((0, 0))  # Fallback

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

def find_junction_centers(junction_pixels):
    """
    Find the center pixel of each junction cluster.
    Uses vectorized ndimage operations instead of Python loops.
    """
    if not np.any(junction_pixels):
        return np.zeros_like(junction_pixels)

    labeled_junctions, n_labels = ndimage.label(junction_pixels)
    if n_labels == 0:
        return np.zeros_like(junction_pixels)

    centers = ndimage.center_of_mass(junction_pixels, labeled_junctions, range(1, n_labels + 1))
    junction_centers = np.zeros_like(junction_pixels)
    for r, c in centers:
        ri, ci = int(round(r)), int(round(c))
        if 0 <= ri < junction_pixels.shape[0] and 0 <= ci < junction_pixels.shape[1]:
            junction_centers[ri, ci] = True
    return junction_centers

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
print(f"  find_junctions: {time.time()-_t0:.1f}s"); _t0 = time.time()

# Label connected components in skeleton
labeled_skeleton = measure.label(skeleton)
props = measure.regionprops(labeled_skeleton)

# Initialize lists for analysis
individuals = []  # Components with no junctions (0 or 1 branch)
networks = []    # Components with junctions (multiple branches)

# Analyze each component
for prop in props:
    # Get the region mask and corresponding junction pixels
    region = prop.image
    region_junctions = junction_pixels[prop.bbox[0]:prop.bbox[2],
                                    prop.bbox[1]:prop.bbox[3]] & region

    # Classify based on junctions
    if np.any(region_junctions):
        networks.append(prop)
    else:
        individuals.append(prop)
print(f"  measure.label + regionprops ({len(props)} components): {time.time()-_t0:.1f}s"); _t0 = time.time()

# Calculate branch statistics using skan
print("Building skeleton graph...")
skeleton_data = csr.Skeleton(skeleton)
branch_data = pd.DataFrame()
branch_data['branch_length'] = skeleton_data.path_lengths() * MICRONS_PER_PIXEL  # Convert to microns immediately
branch_data['network_id'] = -1  # Initialize all branches as unassigned
print(f"  skan.Skeleton + path_lengths ({len(branch_data)} branches): {time.time()-_t0:.1f}s"); _t0 = time.time()

def get_component_centroid(component):
    """Calculate centroid of a region property component"""
    return np.mean(component.coords, axis=0)

def get_network_stats(network_props, branch_data):
    """Calculate network statistics for a complete network"""
    if not network_props:
        return 0, 0, 0  # No networks

    # Get all branches for these networks
    network_indices = [networks.index(prop) for prop in network_props]
    network_branches = branch_data[branch_data['network_id'].isin(network_indices)]

    # Calculate statistics
    n_networks = len(network_props)
    mean_size = len(network_branches) / n_networks if n_networks > 0 else 0
    mean_length = network_branches['branch_length'].mean() if not network_branches.empty else 0

    return n_networks, mean_size, mean_length

# Assign branches to networks using labeled_skeleton (O(branches), not O(networks*branches))
print("Assigning branches to networks...")
label_to_network_idx = {net.label: i for i, net in enumerate(networks)}
branch_network_ids = np.full(len(branch_data), -1, dtype=np.int32)
for branch_idx in range(len(branch_data)):
    path = skeleton_data.path_coordinates(branch_idx)
    pt = path[0].astype(int)
    lbl = labeled_skeleton[pt[0], pt[1]]
    if lbl in label_to_network_idx:
        branch_network_ids[branch_idx] = label_to_network_idx[lbl]
branch_data['network_id'] = branch_network_ids
print(f"  branch-network assignment ({len(networks)} networks): {time.time()-_t0:.1f}s"); _t0 = time.time()

# Keep track of which networks and individuals are associated with nuclei
networks_with_nuclei = set()
individuals_with_nuclei = set()

# Process marked nuclei and generate Excel data
print("Processing nuclei and writing results...")
excel_data = []
for nucleus_id, (contour, area, centroid) in enumerate(zip(nuclei_contours, nuclei_areas, nuclei_centroids), 1):
    # Use pre-calculated nucleus centroid
    nucleus_cx, nucleus_cy = centroid
    nucleus_centroid = np.array([nucleus_cy, nucleus_cx])

    # Find associated networks and individuals
    associated_networks = []
    associated_individuals = []
    max_distance = 120  # Slightly increased from 100 for better detection

    # Check networks
    for i, net in enumerate(networks):
        net_centroid = get_component_centroid(net)
        distance = np.sqrt(np.sum((nucleus_centroid - net_centroid) ** 2))
        if distance < max_distance:
            associated_networks.append(net)
            networks_with_nuclei.add(i)  # Track this network as associated

    # Check individuals
    for i, ind in enumerate(individuals):
        ind_centroid = get_component_centroid(ind)
        distance = np.sqrt(np.sum((nucleus_centroid - ind_centroid) ** 2))
        if distance < max_distance:
            associated_individuals.append(i)
            individuals_with_nuclei.add(i)  # Track this individual as associated

    # Calculate footprint
    footprint = 0
    for net in associated_networks:
        footprint += net.area
    for idx in associated_individuals:
        footprint += individuals[idx].area
    footprint_microns = footprint * (MICRONS_PER_PIXEL ** 2)

    # Get network statistics
    n_networks, mean_network_size, mean_branch_length = get_network_stats(
        associated_networks, branch_data)

    excel_data.append({
        'Nucleus ID': nucleus_id,
        'Nucleus Area (μm²)': area,
        'Mitochondrial Footprint (μm²)': footprint_microns,
        'Individuals': len(associated_individuals),
        'Networks': n_networks,
        'Mean Network Size': mean_network_size,
        'Mean Branch Length (μm)': mean_branch_length
    })
print(f"  process nuclei: {time.time()-_t0:.1f}s"); _t0 = time.time()

# Create and save Excel file
df = pd.DataFrame(excel_data)
excel_path = image_path.rsplit('.', 1)[0] + '_analysis.xlsx'

# Use ExcelWriter with xlsxwriter engine to adjust column widths
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    df.to_excel(writer, index=False, sheet_name='Analysis')
    worksheet = writer.sheets['Analysis']

    # Adjust columns width based on content
    for idx, col in enumerate(df.columns):
        # Get maximum length of column content
        max_length = max(
            df[col].astype(str).apply(len).max(),  # max length of values
            len(str(col))  # length of column name
        )
        # Add a little extra space
        worksheet.set_column(idx, idx, max_length + 2)

print(f"\nAnalysis saved to: {excel_path}")

# --- Visualization ---
print("Generating figures...")
print(f"  Excel write: {time.time()-_t0:.1f}s"); _t0 = time.time()

# Build all figure assets (use coords directly — no full-size masks per component)
_tf = time.time()
junction_centers = find_junction_centers(junction_pixels)
print(f"    find_junction_centers: {time.time()-_tf:.1f}s"); _tf = time.time()
skeleton_rgb = np.zeros((*skeleton.shape, 3))
skeleton_rgb[skeleton] = [0.5, 0.5, 0.5]
for i, network in enumerate(networks):
    if i in networks_with_nuclei:
        r, c = network.coords[:, 0], network.coords[:, 1]
        skeleton_rgb[r, c] = [0, 1, 0]
for i, individual in enumerate(individuals):
    if i in individuals_with_nuclei:
        r, c = individual.coords[:, 0], individual.coords[:, 1]
        skeleton_rgb[r, c] = [0, 0, 1]
skeleton_rgb[junction_centers] = [1, 0, 0]

network_mask = np.zeros(skeleton.shape, dtype=np.uint8)
individual_mask = np.zeros(skeleton.shape, dtype=np.uint8)
for network in networks:
    r, c = network.coords[:, 0], network.coords[:, 1]
    network_mask[r, c] = 1
for individual in individuals:
    r, c = individual.coords[:, 0], individual.coords[:, 1]
    individual_mask[r, c] = 1

print(f"    skeleton_rgb + masks: {time.time()-_tf:.1f}s"); _tf = time.time()
nuclei_img = cv2.cvtColor(img_median.astype(np.float32), cv2.COLOR_GRAY2BGR)
for i, (contour, area, centroid) in enumerate(zip(nuclei_contours, nuclei_areas, nuclei_centroids), 1):
    cv2.drawContours(nuclei_img, [contour], -1, (0, 255, 255), 2)
    cx, cy = centroid
    cv2.putText(nuclei_img, f"{i}: {area:.1f}μm²", (cx - 20, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

# Downsample for display to avoid very slow rendering/saving on large images
MAX_DISPLAY_SIZE = 1024
h, w = original_img.shape[:2]
scale = min(1.0, MAX_DISPLAY_SIZE / max(h, w))
if scale < 1.0:
    new_w, new_h = int(w * scale), int(h * scale)
    orig_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR) if original_img.ndim == 3 else original_img
    disp_img = cv2.resize(orig_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if disp_img.ndim == 2:
        disp_img = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2RGB)
    else:
        disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
    disp_binary = cv2.resize(binary.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    disp_skeleton_rgb = cv2.resize((skeleton_rgb * 255).astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) / 255.0
    disp_skeleton = cv2.resize(skeleton.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    disp_net_ind = cv2.resize((network_mask.astype(np.uint8) + 2 * individual_mask.astype(np.uint8)), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    disp_nuclei = cv2.resize(nuclei_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    disp_gray = cv2.resize(img_median, (new_w, new_h), interpolation=cv2.INTER_AREA)
else:
    disp_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR) if original_img.ndim == 3 else original_img.copy()
    if disp_img.ndim == 2:
        disp_img = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2RGB)
    else:
        disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
    disp_binary = binary
    disp_skeleton_rgb = skeleton_rgb
    disp_skeleton = skeleton
    disp_net_ind = None  # Use separate masks
    disp_nuclei = nuclei_img
    disp_gray = img_median

print(f"    downsample: {time.time()-_tf:.1f}s")
print(f"  build figure assets: {time.time()-_t0:.1f}s"); _t0 = time.time()
fig = plt.figure(figsize=(14, 7))

# Original with overlay
ax1 = plt.subplot(241)
ax1.imshow(disp_img)
ax1.imshow(disp_binary, cmap='magma', alpha=0.2)
ax1.set_title("Original + Binary Mask")
ax1.axis('off')

# Binary
ax2 = plt.subplot(242)
ax2.imshow(disp_binary, cmap='gray')
ax2.set_title("Binary")
ax2.axis('off')

# Skeleton with junctions and nucleus-associated components
ax3 = plt.subplot(243)
ax3.imshow(disp_skeleton_rgb)
ax3.set_title("Skeleton + Junctions\n(Green = Networks, Blue = Individuals\nAssociated with Nuclei)")
ax3.axis('off')

# Simple Skeleton (Black on White)
ax4 = plt.subplot(244)
ax4.imshow(~disp_skeleton, cmap='binary')
ax4.set_title("Skeleton")
ax4.axis('off')

# Networks and Individuals
ax5 = plt.subplot(245)
ax5.imshow(disp_gray, cmap='gray', alpha=0.5)
if disp_net_ind is not None:
    ax5.imshow(disp_net_ind == 1, cmap='Reds', alpha=0.5)
    ax5.imshow(disp_net_ind == 2, cmap='Blues', alpha=0.5)
else:
    ax5.imshow(network_mask, cmap='Reds', alpha=0.5)
    ax5.imshow(individual_mask, cmap='Blues', alpha=0.5)
ax5.set_title(f"Networks ({len(networks)}) and\nIndividuals ({len(individuals)})")
ax5.axis('off')

# Network size distribution
ax6 = plt.subplot(246)
if len(networks) > 0:
    counts, bins, _ = ax6.hist([len(network.coords) for network in networks], bins='auto',
                              color='lightcoral', alpha=0.7, edgecolor='black')
    ax6.axvline(np.mean([len(network.coords) for network in networks]), color='black', linestyle='--',
                label=f'Mean: {np.mean([len(network.coords) for network in networks]):.1f}')
    ax6.set_title("Network Size Distribution\n(Number of Branches per Network)")
    ax6.set_xlabel("Number of Branches")
    ax6.set_ylabel("Frequency")
    ax6.legend()
else:
    ax6.text(0.5, 0.5, "No networks found", ha='center', va='center')
    ax6.set_title("Network Size Distribution")
ax6.axis('on')

# Branch length distribution
ax7 = plt.subplot(247)
if len(branch_data['branch_length']) > 0:
    counts, bins, _ = ax7.hist(branch_data['branch_length'], bins='auto',
                              color='lightgreen', alpha=0.7, edgecolor='black')
    ax7.set_title("Branch Length Distribution")
    ax7.set_xlabel("Length (μm)")
    ax7.set_ylabel("Frequency")
else:
    ax7.text(0.5, 0.5, "No branches found", ha='center', va='center')
    ax7.set_title("Branch Length Distribution")
ax7.axis('on')

# Nuclei Areas Visualization
ax8 = plt.subplot(248)
ax8.imshow(cv2.cvtColor(disp_nuclei.astype(np.uint8), cv2.COLOR_BGR2RGB) / 255.0)
ax8.set_title("Nuclei Areas")
ax8.axis('off')

plt.tight_layout()

# Save the figure before showing it (PNG + 150 DPI for faster save on large images)
figure_path = image_path.rsplit('.', 1)[0] + '_figure.png'
print("Saving figure...")
plt.savefig(figure_path, dpi=150, bbox_inches='tight', format='png')
print(f"  savefig: {time.time()-_t0:.1f}s")
print(f"\n[Total post-nuclei time: {time.time()-_t_start:.1f}s]")
print(f"Figure saved to: {figure_path}")

# Show the figure
plt.show()

# Print results
print(f"\n📄 File: {image_path}")
print("\nMitochondrial Metrics:")
print(f"1. Mitochondrial footprint:")
print(f"   - Area (pixels²): {footprint_area_pixels:.2f}")
print(f"   - Area (μm²): {footprint_area_microns:.2f}")
print(f"   - Percent of image: {footprint_percent:.2f}%")
print(f"2. Number of individuals (puncta and rods): {len(individuals)}")
print(f"3. Number of networks: {len(networks)}")

# Add detection statistics
print("\nAssociation Statistics:")
print(f"4. Networks associated with nuclei: {len(networks_with_nuclei)} of {len(networks)}")
print(f"5. Individuals associated with nuclei: {len(individuals_with_nuclei)} of {len(individuals)}")

print("\nNetwork Size Statistics:")
if len(networks) > 0:
    print(f"6. Mean branches per network: {np.mean([len(network.coords) for network in networks]):.2f}")
    print(f"7. Median branches per network: {np.median([len(network.coords) for network in networks]):.2f}")
    print(f"8. Network size standard deviation: {np.std([len(network.coords) for network in networks]):.2f}")
print("\nNuclei Statistics:")
if nuclei_areas:
    print(f"9. Number of nuclei: {len(nuclei_areas)}")
    print(f"10. Average nucleus area: {np.mean(nuclei_areas):.2f} μm²")
    print(f"11. Total nuclei area: {np.sum(nuclei_areas):.2f} μm²")