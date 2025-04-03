import numpy as np
import cv2
from skimage import io, filters, measure, color, morphology
from skimage.morphology import skeletonize
from skan import csr
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

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

# --- Convert to grayscale if needed ---
if img.ndim == 3:
    print("Image has multiple channels — converting to grayscale.")
    img = color.rgb2gray(img)

# --- Preprocess ---
img = img.astype(np.float32)
img = (img - img.min()) / (img.max() - img.min())  # normalize to [0,1]
threshold = filters.threshold_otsu(img)
binary = img > threshold

# --- Mitochondrial footprint ---
footprint_area_pixels = np.sum(binary)
footprint_percent = 100 * footprint_area_pixels / binary.size

# --- Identify individuals ---
labeled_img = measure.label(binary)
num_individuals = labeled_img.max()

# --- Skeletonize ---
skeleton = skeletonize(binary)

# --- Analyze skeleton ---
skeleton_data = csr.Skeleton(skeleton)
branch_lengths = skeleton_data.path_lengths()
num_branches = len(branch_lengths)

# --- Group branches into networks ---
skeleton_labeled = measure.label(skeleton)
num_networks = skeleton_labeled.max()

# --- Mean network size (branches per network) ---
mean_network_size = num_branches / num_networks if num_networks > 0 else 0

# --- Find endpoints & branchpoints (like Fiji's Analyze Skeleton) ---
def find_branch_points_and_endpoints(skel):
    """Returns coordinates of endpoints and branchpoints in skeleton."""
    from scipy.ndimage import convolve

    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])

    filtered = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
    endpoints = np.argwhere(filtered == 11)  # 1 neighbor + center
    branchpoints = np.argwhere(filtered >= 13)  # 3+ neighbors + center
    return endpoints, branchpoints

endpoints, branchpoints = find_branch_points_and_endpoints(skeleton)

# --- Print results ---
print(f"\n📄 File: {image_path}")
print(f"Mitochondrial footprint (pixels): {footprint_area_pixels}")
print(f"Mitochondrial footprint (% of image): {footprint_percent:.2f}%")
print(f"Number of individuals: {num_individuals}")
print(f"Number of networks: {num_networks}")
print(f"Total number of branches: {num_branches}")
print(f"Mean network size (branches per network): {mean_network_size:.2f}")

# --- Visualization ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, cmap='gray')
ax.imshow(binary, cmap='magma', alpha=0.2)  # Mitochondrial mask (magenta hue)
ax.imshow(skeleton, cmap='Greens', alpha=0.7)  # Skeleton (green)

# Plot endpoints and branchpoints
if len(endpoints) > 0:
    ax.plot(endpoints[:, 1], endpoints[:, 0], 'yo', markersize=4, label='Endpoints')
if len(branchpoints) > 0:
    ax.plot(branchpoints[:, 1], branchpoints[:, 0], 'bo', markersize=4, label='Branchpoints')

ax.set_title("Mitochondria Analysis Preview")
ax.axis('off')
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()