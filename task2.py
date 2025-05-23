import os
import pydicom
import numpy as np
import scipy.ndimage
from collections import deque
import matplotlib.pyplot as plt

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



