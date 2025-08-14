import os
import numpy as np
import nibabel as nib
from skimage.metrics import structural_similarity as ssim

def compute_ssim(voxel_array1: np.ndarray, voxel_array2: np.ndarray) -> np.ndarray:

    # Ensure both volumes have the same shape
    if voxel_array1.shape != voxel_array2.shape:
        raise ValueError(f"Shape mismatch: {voxel_array1.shape} vs {voxel_array2.shape}")

    ssim_scores = np.zeros((voxel_array1.shape[-1],), dtype=np.float32)

    for i in range(voxel_array1.shape[-1]):

        ssim_scores[i] = ssim(voxel_array1[:, :, i], voxel_array2[:, :, i], data_range=255)

    return ssim_scores  # Return the average SSIM across slices

def load_nii(file_path: str) -> np.ndarray:
    """
    Load a 3D medical image from a NIfTI (.nii) file and return its voxel data as a numpy array.

    Parameters:
    -----------
    file_path : str
        Path to the NIfTI file.

    Returns:
    --------
    np.ndarray
        A 3D numpy array representing the loaded voxel data.
    """
    nii_img = nib.load(file_path)  # Load the .nii file
    return nii_img.get_fdata()  # Extract the voxel data as a numpy array

# Directory containing the voxel data
datadir = "G:/ultrasound_data/voxels"

# Define voxel types
voxels = ["real", "nerf", "barf", "barf-no-freq"]

# Load the "real" reference voxel data
real_path = os.path.join(datadir, "real.nii")
real_voxels = load_nii(real_path).astype(np.uint8)

# Compute SSIM scores between "real" and each other voxel dataset
ssim_results = {}

for voxel in voxels:
    if voxel == "real":
        continue  # Skip comparing "real" to itself

    voxel_path = os.path.join(datadir, f"{voxel}.nii")
    
    if not os.path.exists(voxel_path):
        print(f"Warning: {voxel_path} does not exist. Skipping.")
        continue

    voxel_data = load_nii(voxel_path).astype(np.uint8)
    
    # Compute SSIM score
    ssim_scores = compute_ssim(real_voxels, voxel_data)
    ssim_results[voxel] = ssim_scores
    print(f"SSIM between 'real' and '{voxel}': {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
