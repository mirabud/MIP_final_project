import os

import matplotlib
import pydicom
import highdicom
import numpy as np
import scipy
from matplotlib import pyplot as plt, animation



# Load and stack a DICOM series from a folder
def load_dicom_series(folder_path):
    slices = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.dcm'):
            dcm = pydicom.dcmread(os.path.join(folder_path, fname))
            slices.append(dcm)
    # Sort by ImagePositionPatient[2] (Z-axis) or InstanceNumber
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    # Stack into 3D volume
    volume = np.stack([s.pixel_array for s in slices], axis=0)
    return volume, slices

# Visualize an axial slice with optional segmentation overlay
def visualize_slice(ct_volume, seg_volume, slice_idx):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(ct_volume[slice_idx], cmap='gray')
    plt.title(f'CT Slice {slice_idx}')
    
    plt.subplot(1, 2, 2)
    plt.imshow(ct_volume[slice_idx], cmap='gray')
    if seg_volume is not None:
        plt.imshow(seg_volume[slice_idx], cmap='Reds', alpha=0.5)
    plt.title(f'CT + Segmentation Slice {slice_idx}')
    plt.show()

def MIP_sagittal_plane(volume: np.ndarray) -> np.ndarray:
    return np.max(volume, axis=2)

def MIP_coronal_plane(volume: np.ndarray) -> np.ndarray:
    return np.max(volume, axis=1)

def rotate_on_axial_plane(volume: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    return scipy.ndimage.rotate(volume, angle_in_degrees, axes=(1, 2), reshape=False)


def create_animation(volume_ct, volume_seg, mip_func, pixel_len_mm, output_path, n_frames=36):
    angles = np.linspace(0, 360 * (n_frames - 1) / n_frames, num=n_frames)
    projections = []

    fig, ax = plt.subplots()
    for idx, angle in enumerate(angles):
        rotated_ct = rotate_on_axial_plane(volume_ct, angle)
        rotated_seg = rotate_on_axial_plane(volume_seg, angle)
        mip_ct = mip_func(rotated_ct)
        mip_seg = mip_func(rotated_seg)

        combined_img = np.ma.masked_where(mip_seg == 0, mip_seg)
        img = ax.imshow(mip_ct, cmap='gray', animated=True, aspect=pixel_len_mm[0]/pixel_len_mm[1])
        overlay = ax.imshow(combined_img, cmap='Reds', alpha=0.5, animated=True)
        projections.append([img, overlay])

    anim = animation.ArtistAnimation(fig, projections, interval=250, blit=True)

    anim.save(output_path)
    print(f"Animation saved to {output_path}")

    plt.show()  # This will display the animation immediately

    plt.close(fig)

if __name__ == '__main__':
    reference_folder = 'C://Users//Amir//Downloads//RadCTTACEomics_1053-20250409T154823Z-001//RadCTTACEomics_1053//10_AP_Ax2.50mm'
    input_folder = 'C://Users//Amir//Downloads//RadCTTACEomics_1053-20250409T154823Z-001//RadCTTACEomics_1053//20_PP_Ax2.50mm'
    output_dir = 'C://Users//Amir//Desktop//MIP_project//results/MIP'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load reference CT series
    ref_volume, ref_slices = load_dicom_series(reference_folder)
    print(f"Reference volume shape: {ref_volume.shape}")
    
    # Load input segmentation series
    seg_volume, seg_slices = load_dicom_series(input_folder)
    print(f"Segmentation volume shape: {seg_volume.shape}")
    
    # Check matching dimensions
    if ref_volume.shape != seg_volume.shape:
        print("WARNING: Reference and segmentation volumes have mismatched shapes!")
    else:
        print("Reference and segmentation volumes aligned successfully.")

    # Check alignment
    if ref_volume.shape != seg_volume.shape:
        print("WARNING: Reference and segmentation volumes have mismatched shapes!")

    # Normalize CT volume for visualization (optional)
    ref_volume_norm = (ref_volume - np.min(ref_volume)) / (np.max(ref_volume) - np.min(ref_volume))

    # Settings
    n_frames = 36
    angles = np.linspace(0, 360 * (n_frames - 1) / n_frames, num=n_frames)
    projections = []

    pixel_spacing = ref_slices[0].PixelSpacing    # [row_spacing, col_spacing]
    slice_thickness = ref_slices[0].SliceThickness
    pixel_len_mm = [slice_thickness, pixel_spacing[0], pixel_spacing[1]]


    # Sagittal plane animation
    create_animation(
        volume_ct=ref_volume_norm,
        volume_seg=seg_volume,
        mip_func=MIP_sagittal_plane,
        pixel_len_mm=pixel_len_mm,
        output_path=os.path.join(output_dir, 'Tumor_MIP_Rotation_Sagittal.gif'),
        n_frames=36
    )

    # Coronal plane animation
    create_animation(
        volume_ct=ref_volume_norm,
        volume_seg=seg_volume,
        mip_func=MIP_coronal_plane,
        pixel_len_mm=pixel_len_mm,
        output_path=os.path.join(output_dir, 'Tumor_MIP_Rotation_Coronal.gif'),
        n_frames=36
    )
