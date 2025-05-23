import os
import pydicom
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import imageio


def load_dicom_series(folder):
    # Load all DICOM files
    dicoms = [pydicom.dcmread(os.path.join(folder, f)) for f in os.listdir(folder) if f.endswith('.dcm')]
    
    # Sort slices based on Image Position Patient (Z coordinate)
    dicoms.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # Check if all slices belong to the same acquisition
    acquisition_numbers = {d.AcquisitionNumber for d in dicoms if hasattr(d, 'AcquisitionNumber')}
    if len(acquisition_numbers) > 1:
        print(f"⚠️ Warning: Multiple acquisitions found: {acquisition_numbers}")
    else:
        print("✅ All slices are from a single acquisition.")

    # Stack into 3D numpy array
    image_volume = np.stack([d.pixel_array for d in dicoms])
    return dicoms, image_volume

def load_binary_mask(dcm_path):
    dcm = pydicom.dcmread(dcm_path)
    return dcm.pixel_array

def MIP_sagittal_plane(volume: np.ndarray) -> np.ndarray:
    return np.max(volume, axis=2)

def MIP_coronal_plane(volume: np.ndarray) -> np.ndarray:
    return np.max(volume, axis=1)

def rotate_on_axial_plane(volume: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    return scipy.ndimage.rotate(volume, angle_in_degrees, axes=(1, 2), reshape=False)

def create_mip_animation(volume: np.ndarray, plane: str = 'coronal', output_path: str = 'C://Users//Amir//Desktop//MIP_project//results//mip_animation.gif'):
    assert plane in ['sagittal', 'coronal'], "Plane must be 'sagittal' or 'coronal'"

    angles = np.linspace(0, 360, num=16, endpoint=False)
    frames = []

    for angle in angles:
        # Rotate volume around axial axis
        rotated_vol = rotate_on_axial_plane(volume, angle)

        # Apply MIP on selected plane
        if plane == 'coronal':
            mip = MIP_coronal_plane(rotated_vol)
        else:
            mip = MIP_sagittal_plane(rotated_vol)

        # Normalize for display
        mip_normalized = (mip - np.min(mip)) / (np.max(mip) - np.min(mip)) * 255
        mip_normalized = mip_normalized.astype(np.uint8)

        # Plot and save each frame
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(mip_normalized, cmap='gray')
        ax.set_title(f'Angle: {int(angle)}°')
        ax.axis('off')

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close()

    # Save animation as GIF
    imageio.mimsave(output_path, frames, fps=4)
    print(f"✅ Saved animation to: {output_path}")

import numpy as np

def get_bounding_box_and_centroid(mask: np.ndarray):
    # Get coordinates of non-zero (tumor) voxels
    coords = np.argwhere(mask > 0)

    if coords.size == 0:
        raise ValueError("❌ No non-zero voxels found in the mask.")

    # Bounding box: min and max in Z, Y, X
    min_z, min_y, min_x = coords.min(axis=0)
    max_z, max_y, max_x = coords.max(axis=0)

    bbox = {
        'min': (min_z, min_y, min_x),
        'max': (max_z, max_y, max_x),
        'shape': (max_z - min_z + 1, max_y - min_y + 1, max_x - min_x + 1)
    }

    # Centroid (floating-point)
    centroid = coords.mean(axis=0)  # returns (z, y, x)

    return bbox, centroid


reference_folder = 'C://Users//Amir//Downloads//RadCTTACEomics_1053-20250409T154823Z-001//RadCTTACEomics_1053//10_AP_Ax2.50mm'
seg_path = 'C://Users//Amir//Downloads//RadCTTACEomics_1053-20250409T154823Z-001//RadCTTACEomics_1053//10_AP_Ax2.50mm_ManualROI_Tumor.dcm'
files, ref_volume = load_dicom_series(reference_folder)
seg_pixel_array = load_binary_mask(seg_path)
print(seg_pixel_array.shape)
#create_mip_animation(ref_volume, plane='coronal', output_path='mip_coronal.gif')
create_mip_animation(ref_volume, plane='sagittal', output_path='mip_sagittal.gif')
bbox, centroid = get_bounding_box_and_centroid(seg_pixel_array)

print("📦 Bounding Box:")
print(f"  Min: {bbox['min']}")
print(f"  Max: {bbox['max']}")
print(f"  Shape: {bbox['shape']}")

print("\n🎯 Centroid (Z, Y, X):")
print(centroid)

if seg_pixel_array.shape == ref_volume.shape:
    print("CT and segmentation are aligned")
else:
    print("CT and segmentation are not aligned")