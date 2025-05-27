import os
import pydicom
import numpy as np
import scipy.ndimage
from collections import deque
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.segmentation import flood
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_dilation
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                            recall_score, jaccard_score)


# Function to get bounding box and centroid of the tumor
def bbox_and_centroid(mask: np.ndarray):
    # Getting coordinates of non-zero voxels - tumor
    coords = np.argwhere(mask > 0)

    if coords.size == 0:
        raise ValueError("No non-zero voxels found in the mask.")

    # Bounding box
    min_z, min_y, min_x = coords.min(axis=0)
    max_z, max_y, max_x = coords.max(axis=0)

    bbox = {
        'min': (min_z, min_y, min_x),
        'max': (max_z, max_y, max_x),
        'shape': (max_z - min_z + 1, max_y - min_y + 1, max_x - min_x + 1)
    }

    # Centroid
    centroid = coords.mean(axis=0)
    print(f"\n Bounding Box:")
    print(f"  Min: {bbox['min']}")
    print(f"  Max: {bbox['max']}")
    print(f"  Shape: {bbox['shape']}")

    print("\n Centroid (Z, Y, X):")
    print(centroid)
    return bbox, centroid


# Function to apply Gaussian Blur on the CT slices
def preprocessing(ct_volume, sigma=0.5):
    return gaussian_filter(ct_volume, sigma=sigma)


# Function to apply Manual Region Growing segmentation
def region_growing_manual(ct_volume: np.ndarray, seed: tuple, bbox: dict = None, threshold: float = 5.0) -> np.ndarray:

    # Getting the seed intensity by the CT shape
    z_dim, y_dim, x_dim = ct_volume.shape
    seed_z, seed_y, seed_x = map(int, seed)
    seed_intensity = ct_volume[seed_z, seed_y, seed_x]
    
    # Getting bounding box limits
    if bbox:
        z_min, y_min, x_min = bbox['min']
        z_max, y_max, x_max = bbox['max']
    else:
        z_min, y_min, x_min, z_max, y_max, x_max = 0, 0, 0, z_dim-1, y_dim-1, x_dim-1

    # Creating a null array of the shape of CT volume
    visited = np.zeros_like(ct_volume, dtype=bool)
    # Same with mask
    mask = np.zeros_like(ct_volume, dtype=np.uint8)
    
    # A library that allows data structures to add and remove elements from both ends
    queue = deque()
    queue.append((seed_z, seed_y, seed_x))
    visited[seed_z, seed_y, seed_x] = True

    while queue:
        z, y, x = queue.popleft()
        
        # Skipping points outside the bounding box
        if not (z_min <= z <= z_max and y_min <= y <= y_max and x_min <= x <= x_max):
            continue
        
        current_intensity = ct_volume[z, y, x]
        
        # Checking if intensity is within threshold
        if abs(current_intensity - seed_intensity) <= threshold:
            mask[z, y, x] = 1
            
            # 26-connectivity neighbors (all adjacent points in 3D)
            for dz, dy, dx in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1),
                                 (-1,-1,0),(-1,1,0),(1,-1,0),(1,1,0),
                                 (0,-1,-1),(0,-1,1),(0,1,-1),(0,1,1),
                                 (-1,0,-1),(-1,0,1),(1,0,-1),(1,0,1)]:
                nz, ny, nx = z + dz, y + dy, x + dx
                if 0 <= nz < z_dim and 0 <= ny < y_dim and 0 <= nx < x_dim:
                    if not visited[nz, ny, nx]:
                        visited[nz, ny, nx] = True
                        queue.append((nz, ny, nx))
    print("Segmented voxels count - Manual:", np.count_nonzero(mask))
    
    return mask

# Function to apply Region Growing segmentation with Skimage FloodFill
def region_growing_floodfill(ct_volume, seed_point, bbox, tolerance=36, iterations=1):

    seed_z, seed_y, seed_x = map(int, seed_point) 
    z_min, y_min, x_min = map(int, bbox['min'])  
    z_max, y_max, x_max = map(int, bbox['max'])
    
    # Checking if seeds are within the bounding box
    if not (z_min <= seed_z < z_max and 
            y_min <= seed_y < y_max and 
            x_min <= seed_x < x_max):
        raise ValueError("Seed point is outside the bounding box")
    
    # Extracting ROI and ensuring it's a copy
    roi = np.array(ct_volume[z_min:z_max, y_min:y_max, x_min:x_max], copy=True)
    
    # Calculating relative seed position within ROI
    seed_rel = (seed_z - z_min, seed_y - y_min, seed_x - x_min)
    
    # Getting seed value (with bounds check)
    try:
        seed_value = roi[seed_rel]
    except IndexError as e:
        raise IndexError(f"Relative seed position {seed_rel} is invalid for ROI shape {roi.shape}") from e
    
    mask = flood(
        roi, 
        seed_point=seed_rel,
        tolerance=tolerance,
        connectivity=3 # FloodFill has only 6 connectivity but below we manually go over 26
    )
    
    # Iterative dilation to 26
    if iterations > 0:
        struct = np.ones((3,3,3), dtype=bool)
        mask = binary_dilation(mask, structure=struct, iterations=iterations)
    
    # Keeping only largest component
    labeled = label(mask)
    if labeled.max() == 0:
        return np.zeros_like(ct_volume, dtype=np.uint8)
    
    component_sizes = np.bincount(labeled.flat)[1:]
    largest_component = np.argmax(component_sizes) + 1
    roi_mask = (labeled == largest_component)
    
    # Mapping back to original volume
    full_mask = np.zeros_like(ct_volume, dtype=np.uint8)
    full_mask[z_min:z_max, y_min:y_max, x_min:x_max] = roi_mask
    print("Segmented voxels count - FloodFill:", np.count_nonzero(full_mask))
    
    return full_mask


# Function to apply Watershed segmentation
def segment_tumor_watershed(ct_volume, bbox, seed_point):

    seed_z, seed_y, seed_x = map(int, seed_point)
    z_min, y_min, x_min = bbox['min']
    z_max, y_max, x_max = bbox['max']
    roi = ct_volume[z_min:z_max, y_min:y_max, x_min:x_max]
    
    # Computing distance transform
    threshold = np.percentile(roi, 2.7)
    binary = roi > threshold
    distance = ndi.distance_transform_edt(binary)
    
    # Finding markers
    coords = peak_local_max(distance, footprint=np.ones((3,3,3)), labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = label(mask)
    
    # Applying watershed
    labels = watershed(-distance, markers, mask=binary)
    
    # Selecting region containing seed point
    seed_rel = (seed_z-z_min, seed_y-y_min, seed_x-x_min)
    target_label = labels[seed_rel]
    tumor_mask = (labels == target_label)
    
    full_mask = np.zeros_like(ct_volume, dtype=np.uint8)
    full_mask[z_min:z_max, y_min:y_max, x_min:x_max] = tumor_mask
    print("Segmented voxels count - Watershed:", np.count_nonzero(full_mask))
    return full_mask


# Function to visualize MIP with segmented tumor
def show_mip_views(ct_volume, segmentation_mask):

    # MIPs of the CT volume
    mip_axial = np.max(ct_volume, axis=0)
    mip_sagittal = np.max(ct_volume, axis=1)
    mip_coronal = np.max(ct_volume, axis=2)
    
    # MIPs of the segmentation
    seg_mip_axial = np.max(segmentation_mask, axis=0)
    seg_mip_sagittal = np.max(segmentation_mask, axis=1)
    seg_mip_coronal = np.max(segmentation_mask, axis=2)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Axial MIP
    axes[0].imshow(mip_axial, cmap='gray')
    axes[0].imshow(np.ma.masked_where(seg_mip_axial == 0, seg_mip_axial), 
                  cmap='Reds', alpha=0.7, vmin=0, vmax=1)
    axes[0].set_title('Axial MIP')
    
    # Sagittal MIP
    axes[1].imshow(mip_sagittal.T, cmap='gray')
    axes[1].imshow(np.ma.masked_where(seg_mip_sagittal.T == 0, seg_mip_sagittal.T), 
                  cmap='Reds', alpha=0.7, vmin=0, vmax=1)
    axes[1].set_title('Sagittal MIP')
    
    # Coronal MIP
    axes[2].imshow(mip_coronal, cmap='gray')
    axes[2].imshow(np.ma.masked_where(seg_mip_coronal == 0, seg_mip_coronal), 
                  cmap='Reds', alpha=0.7, vmin=0, vmax=1)
    axes[2].set_title('Coronal MIP')
    
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# Function to calculate metrics
def evaluate_methods(gt_mask, methods_dict):

    results = {}
    
    for method_name, pred_mask in methods_dict.items():
        # Settings masks as binary and in same shape
        pred_binary = (pred_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        assert pred_binary.shape == gt_binary.shape, "Masks must have same dimensions"
        
        # Calculate metrics
        results[method_name] = {
            'Dice': f1_score(gt_binary.flatten(), pred_binary.flatten()),
            'IoU': jaccard_score(gt_binary.flatten(), pred_binary.flatten()),
            'Precision': precision_score(gt_binary.flatten(), pred_binary.flatten()),
            'Recall': recall_score(gt_binary.flatten(), pred_binary.flatten()),
            'F1': f1_score(gt_binary.flatten(), pred_binary.flatten()),
            'Accuracy': accuracy_score(gt_binary.flatten(), pred_binary.flatten()),
            'Specificity': recall_score(gt_binary.flatten(), pred_binary.flatten(), pos_label=0)
        }
    
    return results

# Function to plot visual comparison of metrics
def plot_metrics_comparison(results):
    metrics = list(results[next(iter(results))].keys())
    methods = list(results.keys())
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, method in enumerate(methods):
        values = [results[method][m] for m in metrics]
        ax.bar(x + i*width, values, width, label=method)
    
    ax.set_ylabel('Scores')
    ax.set_title('Segmentation Method Comparison')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()


# Function to show slice by slice comparison
def show_comparison_slices(ct_volume, gt_mask, methods_dict, slice_idx):

    plt.figure(figsize=(10, 5))
    
    # Ground truth
    plt.subplot(1, len(methods_dict)+1, 1)
    plt.imshow(ct_volume[slice_idx], cmap='gray')
    plt.imshow(gt_mask[slice_idx], cmap='Reds', alpha=0.3)
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Each method
    for i, (name, mask) in enumerate(methods_dict.items(), 2):
        plt.subplot(1, len(methods_dict)+1, i)
        plt.imshow(ct_volume[slice_idx], cmap='gray')
        plt.imshow(mask[slice_idx], cmap='Blues', alpha=0.3)
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()