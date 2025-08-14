import argparse
import os

import numpy as np


from utils.nerf import load_experiment_args
from utils.pose import convert_to_homogeneous, load_and_refine_poses

from skimage.metrics import structural_similarity as ssim


def compute_voxel_grid_limits(poses, H, W, sh, sw, margin=0.01):
    """
    Determines the 3D bounding box for all transformed slices.
    Ensures all compounded volumes align in the same space.

    Args:
        poses (np.ndarray): Array of (N, 4, 4) transformation matrices.
        H (int): Image height.
        W (int): Image width.
        sh (float): Scaling factor for height.
        sw (float): Scaling factor for width.
        margin (float): Additional padding for safety.

    Returns:
        voxel_origin (np.ndarray): Minimum [x, y, z] coordinate of the grid.
        voxel_size (float): Grid resolution.
        grid_shape (tuple): Shape (Dx, Dy, Dz) of the voxel volume.
    """

    corners = []
    for pose in poses:
        t = pose[:3, -1]
        R = pose[:3, :3]

        y = np.array([-H / 2, -H / 2, H / 2, H / 2], dtype=np.float32) * sh * -1
        x = np.array([-W / 2, W / 2, -W / 2, W / 2], dtype=np.float32) * sw
        z = np.zeros_like(x)

        base = np.stack([x, y, z], axis=1)

        rotated = (R @ base.T).T
        shifted = rotated + t

        corners.append(shifted)

    corners = np.concatenate(corners, axis=0)
    min_corner = corners.min(axis=0) - margin
    max_corner = corners.max(axis=0) + margin

    # Define voxel grid resolution
    voxel_size = min(sh, sw) * 16  # Use 16x pixel size
    grid_shape = np.ceil((max_corner - min_corner) / voxel_size).astype(int)

    return min_corner, voxel_size, grid_shape


def images_to_voxel_grid(images, poses, H, W, sh, sw, voxel_size=0.01):
    """
    Converts images & poses into a 3D voxel grid representation.
    """
    N = len(images)
    voxel_dict = {}

    for i in range(N):
        img = images[i]
        pose = poses[i]

        y, x = np.meshgrid(
            np.linspace(-W / 2, W / 2, W) * sw, np.linspace(-H / 2, H / 2, H) * sh * -1
        )
        z = np.zeros_like(x)
        img_coords = np.stack([x, y, z, np.ones_like(x)], axis=-1).reshape(-1, 4)

        world_coords = (pose @ img_coords.T).T[:, :3]
        voxel_indices = np.floor(world_coords / voxel_size).astype(int)

        for idx, val in zip(map(tuple, voxel_indices), img.flatten()):
            if val > 10:  # Threshold to remove noise
                voxel_dict[idx] = max(voxel_dict.get(idx, 0), val)

    return voxel_dict


def compute_ssim_between_volumes(
    images1, poses1, images2, poses2, H, W, sh, sw, voxel_size=0.01
):
    """
    Computes SSIM between two 3D voxel volumes.
    """
    vol1 = images_to_voxel_grid(images1, poses1, H, W, sh, sw, voxel_size)
    vol2 = images_to_voxel_grid(images2, poses2, H, W, sh, sw, voxel_size)

    common_keys = set(vol1.keys()) | set(vol2.keys())
    v1 = np.array([vol1.get(k, 0) for k in common_keys])
    v2 = np.array([vol2.get(k, 0) for k in common_keys])

    return ssim(v1, v2)


def main(args):

    exp_args = load_experiment_args(args.args_file)

    datadir = (
        exp_args["datadir"]
        .replace("/home/guests/{NAME}", "G:")
        .replace("/home/guests/{NAME}", "G:")
    )
    pose_path = (
        exp_args["pose_path"]
        .replace("/home/guests/{NAME}", "G:")
        .replace("/home/guests/{NAME}", "G:")
    )
    weights_dir = os.path.join("logs", exp_args["expname"])
    images_path = os.path.join(datadir, "images.npy")

    iterations = [100000, 200000, 300000, 400000, 500000]

    poses_data = load_and_refine_poses(datadir, pose_path, weights_dir, iterations)

    # Load images
    images = np.load(images_path)  # Load the ultrasound images (N, H, W)
    images = images.transpose(0, 2, 1)

    scaling = 0.001
    probe_depth = float(exp_args["probe_depth"]) * scaling
    probe_width = float(exp_args["probe_width"]) * scaling

    # Define scaling factors (assuming these are known)
    H, W = images.shape[1], images.shape[2]

    sw = probe_depth / float(W)
    sh = probe_width / float(H)

    poses_to_vis = {
        -1: poses_data["original"],
        0: poses_data["noisy"],
        **poses_data["refined"],
    }

    for iter_no, poses in poses_to_vis.items():
        print(f"Visualizing ultrasound slices for iteration {iter_no}...")

        poses = convert_to_homogeneous(poses)

        ssim_value = compute_ssim_between_volumes(
            images, convert_to_homogeneous(poses_to_vis[-1]), images, poses, H, W, sh, sw
        )
        print(f"SSIM between volumes: {ssim_value}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Applies 3D compouding to convert freehand ultrasound data into voxel based format. Then calculated SSIM to measure tracking improvement."
    )
    parser.add_argument(
        "args_file",
        type=str,
        help="File or directory containing the arguments for the experiment",
    )

    args = parser.parse_args()

    main(args)
