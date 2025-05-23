import os
import pydicom
import numpy as np
import scipy.ndimage
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Function to read, sort, and save the CT slices
def load_images(folder):
    dicoms = [pydicom.dcmread(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith('.dcm')]
    
    # Sorting slices based on Image Position Patient (Z coordinate)
    dicoms.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # Checking if all slices belong to the same acquisition
    acquisition_numbers = {d.AcquisitionNumber for d in dicoms if hasattr(d, 'AcquisitionNumber')}
    if len(acquisition_numbers) > 1:
        # If not, getting only those with the same acquisition
        print(f"Multiple acquisitions found: {acquisition_numbers}")
        acq_number = min(d.AcquisitionNumber for d in dicoms)
        dicoms = [d for d in dicoms if d.AcquisitionNumber == acq_number]
    else:
        print("All slices are from a single acquisition.")

    # Stack into 3D numpy array (getting volume)
    image_volume = np.stack([d.pixel_array for d in dicoms])

    z_positions = [float(d.ImagePositionPatient[2]) for d in dicoms]
    return dicoms, image_volume, z_positions


# Function to read, sort, and save the segmentation masks
def load_mask(dcm_path):
    dcm = pydicom.dcmread(dcm_path)
    # Getting the binary mask
    mask = dcm.pixel_array
    # Getting the z positions according to Image Position Patient dcm header to use for visualization
    frame_z_positions = [
    float(frame.PlanePositionSequence[0].ImagePositionPatient[2])
    for frame in dcm.PerFrameFunctionalGroupsSequence
]
    return mask, frame_z_positions


# Function to reorder the masks
def reorder_masks(ct_volume, mask, ct_z_positions, z_positions):
    # Creating an empty array to later reorder the CT slices and masks according to patient position
    mask_volume = np.zeros_like(ct_volume, dtype=bool)
    ct_z_array = np.array(ct_z_positions)

    # Reordering the mask based on closest Z positions between CT and segmentation
    for f, seg_z in enumerate(z_positions):
        slice_idx = int(np.argmin(np.abs(ct_z_array - seg_z)))
        
        # Checking if the slice index is within bounds
        if slice_idx < mask_volume.shape[0]:
            mask_volume[slice_idx, :, :] = mask[f] > 0

    return mask_volume


# Calculating MIP and rotating the CT and masks (functions taken from lesson activity #3)
# Using coronal plane for animation
def MIP_coronal_plane(volume: np.ndarray) -> np.ndarray:
    return np.max(volume, axis=1)

def rotate_on_axial_plane(volume: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    return scipy.ndimage.rotate(volume, angle_in_degrees, axes=(1, 2), reshape=False)


# Function to visualize the slice with liver and tumor masks
def slice_overlay(ct_volume, liver_mask, tumor_mask, slice_idx=None):
    
    if slice_idx is None:
        # Taking the middle slice
        slice_idx = ct_volume.shape[0] // 2 
    
    ct_slice = ct_volume[slice_idx]
    liver_slice = liver_mask[slice_idx]
    tumor_slice = tumor_mask[slice_idx]

    plt.figure(figsize=(5, 5))
    plt.imshow(ct_slice, cmap='gray')
    plt.imshow(liver_slice, cmap='Greens', alpha=0.4)
    plt.imshow(tumor_slice, cmap='Reds', alpha=0.5) 
    plt.title(f'CT Slice {slice_idx} with Liver and Tumor masks')
    plt.axis('off')
    plt.show()


# Rotating the slices with masks, calculating MIP and saving them to 16 frames for animation
def calculate_mip_coronal(ct_volume: np.ndarray, tumor_mask_volume: np.ndarray, liver_mask_volume: np.ndarray):

    angles = np.linspace(0, 360, num=16, endpoint=False)
    frames = []

    for angle in angles:
        rotated_ct = rotate_on_axial_plane(ct_volume, angle)
        rotated_mask_tumor = rotate_on_axial_plane(tumor_mask_volume, angle)
        rotated_mask_liver = rotate_on_axial_plane(liver_mask_volume, angle)
        
        mip_ct   = MIP_coronal_plane(rotated_ct)
        mip_mask_tumor = MIP_coronal_plane(rotated_mask_tumor) > 0
        mip_mask_liver = MIP_coronal_plane(rotated_mask_liver) > 0
        frames.append((mip_ct, mip_mask_tumor, mip_mask_liver))

    return frames


