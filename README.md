# Medical Image Processing Final Project
### Mira Budenova

There are two goals of this project:
1. Loading and visualizing DICOM files, where the objectives are:
- Visualize them with the third-party software - 3D Slicer;
- Load, read, process, and sort CT files and its segmentations. Additionally, rearrange them using DICOM headers.
- Create an animation, showing a rotating Maximum Intensity Projection (MIP) on the coronal-sagittal planes, visualizing the masks. 
2. 3-D Image Segmentation, where the objectives are:
- Extract the bounding box and centroid of a tumor mask;
- Create a semi-automatic segmentation algorithm that only uses the CT image, and either the bounding box or the centroid of the tumor.
- Visualize both the provided Tumor mask and the segmented Tumor mask on the image, and assess the correctness of the algorithm, numerically and visually.

The project is focused on the tumor in the liver of the patient. There is folder containing the CT slices, and 2 manually segmented masks of the tumor and the liver.

### Structure of the directory

The directory contains of 2 helper files: task1.py and task2.py, containing all functions for task 1 and task 2 respectively, and the main Jupiter file, containing the implementations of the helper documents, and "results" folder, containing the PNG frames and the animation of rotated MIP on coronal plane, visualizing the masks.
