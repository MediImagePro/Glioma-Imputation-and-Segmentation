import os
import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage

def nii_to_png(nii_path, output_dir, slice_thickness=1, axis=2, threshold=1, min_nonzero_ratio=0.15):
    """
    Convert NIfTI data to PNG images by slicing, save only the slices with sufficient information.

    Parameters:
    - nii_path: Path to the NIfTI file.
    - output_dir: Directory to save the PNG images.
    - slice_thickness: Thickness of each slice (in terms of number of slices).
    - axis: Axis to slice along (0, 1, or 2).
    - threshold: Pixel intensity threshold to differentiate foreground from background.
    - min_nonzero_ratio: Minimum ratio of nonzero pixels to consider a slice as containing sufficient information.
    """
    # Load the NIfTI file
    img = nib.load(nii_path)
    data = img.get_fdata()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Slice and save as PNG
    for i in range(0, data.shape[axis], slice_thickness):
        if axis == 0:
            slice_data = data[i, :, :]
        elif axis == 1:
            slice_data = data[:, i, :]
        else:  # axis == 2
            slice_data = data[:, :, i]

        # Apply threshold to set background pixels to black
        slice_data[slice_data <= threshold] = 0

        # Morphological operations to remove background noise
        slice_data = ndimage.binary_opening(slice_data, structure=np.ones((3,3))).astype(np.uint8)

        # Check if the slice contains sufficient information
        nonzero_ratio = np.count_nonzero(slice_data) / slice_data.size
        if nonzero_ratio > min_nonzero_ratio:
            # Normalize the slice to 0-255
            if np.max(slice_data) != 0:  # Prevent division by zero
                slice_data = 255 * (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
            slice_img = Image.fromarray(slice_data.astype(np.uint8))

            # Save the slice
            slice_img.save(os.path.join(output_dir, f"slice_{i}.png"))

if __name__ == "__main__":
    nii_path = r"/test.nii.gz"
    output_dir = r"dataroot"
    threshold = 1  # This is an example threshold value
    nii_to_png(nii_path, output_dir, threshold=threshold)
