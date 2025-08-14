import argparse
import json
import os

import numpy as np
import SimpleITK as sitk

from utils.pose import apply_global_registration, reflect_image_poses


def rotate_image_poses(poses: np.ndarray, axis: str, angle: float) -> np.ndarray:
    """
    Rotates a batch of 4x4 pose matrices in place, keeping their global positions fixed
    while changing their orientations as if the images are in the XZ plane.

    Parameters:
        poses (np.ndarray): An (N, 4, 4) array of pose matrices.
        axis (str): The axis to rotate around ('x', 'y', or 'z').
        angle (float): The angle in degrees to rotate.

    Returns:
        np.ndarray: The updated (N, 4, 4) array with rotation applied in local space.
    """
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("Poses must be an (N, 4, 4) array")

    # Convert degrees to radians
    theta = np.radians(angle)

    # Define rotation matrices (applied in local coordinates)
    if axis.lower() == "x":
        R = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )
    elif axis.lower() == "y":
        R = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )
    elif axis.lower() == "z":
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    # Apply rotation to all poses
    new_poses = poses.copy()
    new_poses[:, :3, :3] = poses[:, :3, :3] @ R
    return new_poses


transform_template = "Seq_Frame{:04d}_ImageToReferenceTransform"


def poses_from_mha_image(image):
    """Extracts the tracking information from the given MHA image.

    Args:
        image (SimpleITK.Image): The MHA image.

    Returns:
        np.ndarray: The transformation matrices for each slice.
    """

    # Get number of slices
    num_slices = image.GetDepth()

    # Get tracking matrices as string
    transform_lines = [
        image.GetMetaData(transform_template.format(i)) for i in range(num_slices)
    ]

    # Convert to transformation matrices
    transform_matrix = np.stack(
        [
            np.array(list(map(float, line.split()))).reshape(4, 4)
            for line in transform_lines
        ],
        axis=0,
    )

    transform_matrix = rotate_image_poses(transform_matrix, "z", -90)
    transform_matrix = reflect_image_poses(transform_matrix, "x")

    return transform_matrix


def read_mha_file(file_path):
    # Read the MHA file.
    image = sitk.ReadImage(file_path)

    # Get the image data.
    image_data = sitk.GetArrayFromImage(image)

    # Get the image spacing.
    image_spacing = image.GetSpacing()

    # Get poses
    poses = poses_from_mha_image(image)

    print(
        f"For file {os.path.basename(file_path)}: Probe width: {image_spacing[0] * image_data.shape[2]}, probe depth: {image_spacing[1] * image_data.shape[1]}"
    )

    return image_data, poses, image_spacing


def main(args):
    # Read the MHA file.
    image_data, poses, image_spacing = read_mha_file(args.input)

    if args.reg:
        with open(args.reg, "r") as f:
            reg_params = json.load(f)

        poses = apply_global_registration(poses, **reg_params)

    # Create the output directory.
    os.makedirs(args.output, exist_ok=True)

    # Create images directory
    images_dir = os.path.join(args.output, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Save iamges and poses as numpy files
    np.save(os.path.join(args.output, "images.npy"), image_data)
    np.save(os.path.join(args.output, "poses.npy"), poses)

    # Save each slice as a separate PNG image
    # for i, slice_data in enumerate(image_data):
    #     slice_image = Image.fromarray(slice_data)
    #     slice_image.save(os.path.join(images_dir, f"{i}.png"))

    # Save spacing
    with open(os.path.join(args.output, "spacing.txt"), "w") as f:
        f.write(f"{image_spacing[0]} {image_spacing[1]} {image_spacing[2]}")

    print(f"Saved images and poses to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts a MHA file to an UltraNerf dataset."
    )
    parser.add_argument("input", type=str, help="The input MHA file.")
    parser.add_argument(
        "--reg",
        type=str,
        help="JSON File containing arbitrary registration parameters if necessary. This is mainly to be in the same space with UltraNERF.",
    )
    parser.add_argument("output", type=str, help="The output directory.")

    args = parser.parse_args()

    main(args)
