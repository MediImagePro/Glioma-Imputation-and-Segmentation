import os
import numpy as np
import nibabel as nib
from PIL import Image


def pngs_to_nifti(png_dir, output_file):
    """
    Convert a series of PNG images into a single NIfTI file.

    Parameters:
    - png_dir: Directory containing the PNG images.
    - output_file: Path to the output NIfTI file.
    """
    # Find all PNG files and sort by the slice number extracted from the file name
    png_files = [f for f in os.listdir(png_dir) if f.endswith('_fake_B.png')]
    png_files.sort(key=lambda x: int(x.split('_')[1]))

    # Load the first image to get dimensions
    sample_img = Image.open(os.path.join(png_dir, png_files[0])).convert('L')
    width, height = sample_img.size

    # Create a 3D array to hold the pixel values from all images
    img_data = np.zeros((height, width, len(png_files)), dtype=np.int16)

    # Load each image, convert to grayscale, and add it to the array
    for i, file_name in enumerate(png_files):
        img_path = os.path.join(png_dir, file_name)
        img = Image.open(img_path).convert('L')  # Convert the image to grayscale
        img_data[:, :, i] = np.array(img)

    # Convert the numpy array to a NIfTI image
    nifti_img = nib.Nifti1Image(img_data, affine=np.eye(4))

    # Save the NIfTI image
    nib.save(nifti_img, output_file)


if __name__ == "__main__":
    png_dir = r""  # PNG图片目录
    output_file = "test.nii.gz"  # 输出文件
    pngs_to_nifti(png_dir, output_file)
