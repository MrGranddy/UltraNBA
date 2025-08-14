import argparse
import os

import nibabel as nib
import numpy as np
import torch

from camera import Lie, Pose
from utils.nerf import load_experiment_args
from utils.pose import (
    convert_to_homogeneous,
    flatten_poses,
    load_pose_data,
    save_to_csv,
)


def refine_poses(
    noisy_poses: torch.Tensor, pose_refine: torch.Tensor, lie: Lie, pose: Pose
) -> torch.Tensor:
    """
    Apply refinement to noisy camera poses using Lie algebra-based transformations.

    Args:
        noisy_poses (torch.Tensor): The initial noisy poses.
        pose_refine (torch.Tensor): Refinement transformations.
        lie (Lie): Lie algebra transformation utilities.
        pose (Pose): Pose composition utilities.

    Returns:
        torch.Tensor: Refined poses.
    """
    refined_poses = torch.zeros_like(noisy_poses)
    for i in range(noisy_poses.shape[0]):
        refine_mat = lie.se3_to_SE3(pose_refine[i])
        refine_mat[:, 3] *= 0.001
        refined_poses[i] = pose.compose([refine_mat, noisy_poses[i]])
    return refined_poses


def save_images_as_nifti(images_path: str, output_filename: str) -> None:
    """
    Load images, transform their shape, and save them as a NIfTI file.

    Args:
        images_path (str): Path to the numpy file containing images.
        output_filename (str): Name of the output NIfTI file.
    """
    images = np.load(images_path)
    images = np.transpose(images, (1, 2, 0))
    nii_img = nib.Nifti1Image(images, affine=np.eye(4))
    nib.save(nii_img, output_filename)


def save_images_as_nifti_from_array(images: np.ndarray, output_filename: str) -> None:
    """
    Load images, transform their shape, and save them as a NIfTI file.

    Args:
        images_path (str): Path to the numpy file containing images.
        output_filename (str): Name of the output NIfTI file.
    """
    images = np.transpose(images, (1, 2, 0))
    nii_img = nib.Nifti1Image(images, affine=np.eye(4))
    nib.save(nii_img, output_filename)


def preprocess_poses(poses):

    poses[:, :3, 3] *= 0.001
    flip_y = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    poses = np.stack([pose @ flip_y for pose in poses])

    return poses


def postprocess_poses(poses):

    flip_y = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    poses = np.stack([pose @ flip_y for pose in poses])

    poses[:, :3, 3] /= 0.001

    return poses


RESULTS_DIR = "imfusion_results"


def main():
    """Main function to process camera pose refinement and save results."""
    parser = argparse.ArgumentParser(
        description="Calculate adjusted BARF camera errors"
    )
    parser.add_argument(
        "args_file",
        type=str,
        help="File or directory containing the arguments for the experiment",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Image array path if for example you want to create the volume from rendered images.",
    )
    args = parser.parse_args()

    if args.args_file.endswith(".txt"):  # Trained experiment provided

        # Load experiment arguments
        exp_args = load_experiment_args(args.args_file)
        expname = exp_args["expname"]

        pose_path = (
            exp_args["pose_path"]
            .replace("/home/guests/{NAME}", "G:")
            .replace("/home/guests/{NAME}", "G:")
        )
        datadir = (
            exp_args["datadir"]
            .replace("/home/guests/{NAME}", "G:")
            .replace("/home/guests/{NAME}", "G:")
        )

        # Create structured directory for storing results
        exp_dir = os.path.join(RESULTS_DIR, expname)
        os.makedirs(exp_dir, exist_ok=True)

        # Load camera poses
        org_poses = load_pose_data(os.path.join(datadir, "poses.npy")).cpu().numpy()
        if pose_path and pose_path != "None":
            noisy_poses = load_pose_data(pose_path).cpu().numpy()
        else:
            noisy_poses = org_poses.copy()

        # Preprocess poses
        noisy_poses = preprocess_poses(noisy_poses)
        org_poses = preprocess_poses(org_poses)

        # Load model checkpoint for pose refinement
        weights_path = os.path.join(
            exp_args["basedir"], expname, f"{exp_args['n_iters']}.tar"
        )
        ckpt = torch.load(
            weights_path, map_location=torch.device("cpu"), weights_only=True
        )
        pose_refine = ckpt["pose_refine_state_dict"]["se3_refine.weight"].cpu()

        # Refine poses
        pose = Pose()
        lie = Lie()
        refined_poses = (
            refine_poses(torch.tensor(noisy_poses).float(), pose_refine, lie, pose)
            .cpu()
            .numpy()
        )

        # Convert and save poses
        org_poses = convert_to_homogeneous(org_poses)
        noisy_poses = convert_to_homogeneous(noisy_poses)
        refined_poses = convert_to_homogeneous(refined_poses)

        # Postprocess poses
        org_poses = postprocess_poses(org_poses)
        noisy_poses = postprocess_poses(noisy_poses)
        refined_poses = postprocess_poses(refined_poses)

        save_to_csv(
            os.path.join(exp_dir, "original_poses.csv"), flatten_poses(org_poses)
        )
        save_to_csv(
            os.path.join(exp_dir, "noisy_poses.csv"), flatten_poses(noisy_poses)
        )
        save_to_csv(
            os.path.join(exp_dir, "refined_poses.csv"), flatten_poses(refined_poses)
        )

        if args.images:

            images = np.load(args.images)
            print(images.shape)

            save_images_as_nifti_from_array(
                images,
                os.path.join(exp_dir, "output.nii"),
            )
        else:
            # Process and save images as NIfTI
            save_images_as_nifti(
                os.path.join(datadir, "images.npy"),
                os.path.join(exp_dir, "output.nii"),
            )

        print(f"Processing Done. Experiment is saved to {exp_dir}.")

    elif os.path.isdir(args.args_file):  # Raw data directory provided

        data_dir = args.args_file
        images_path = os.path.join(data_dir, "images.npy")
        poses_path = os.path.join(data_dir, "poses.npy")

        # Ensure paths exist
        if not os.path.exists(images_path) or not os.path.exists(poses_path):
            raise FileNotFoundError(
                "Both images.npy and poses.npy must exist in the given directory."
            )

        exp_dir = os.path.join(
            RESULTS_DIR, os.path.basename(os.path.normpath(data_dir))
        )
        os.makedirs(exp_dir, exist_ok=True)

        # Load poses
        org_poses = load_pose_data(
            poses_path
        )  # Assuming original poses are the same file

        # Convert and save poses
        org_poses = convert_to_homogeneous(org_poses.cpu().numpy())

        save_to_csv(
            os.path.join(exp_dir, "original_poses.csv"), flatten_poses(org_poses)
        )

        noisy_path = os.path.join(data_dir, "noisy_poses")
        if os.path.isdir(noisy_path):

            for noisy_pose_name in os.listdir(noisy_path):
                noisy_pose = load_pose_data(os.path.join(noisy_path, noisy_pose_name))
                noisy_pose = convert_to_homogeneous(noisy_pose.cpu().numpy())
                noisy_name = ".".join(noisy_pose_name.split(".")[:-1])
                save_to_csv(
                    os.path.join(exp_dir, f"{noisy_name}.csv"),
                    flatten_poses(noisy_pose),
                )

        # Process and save images as NIfTI
        save_images_as_nifti(images_path, os.path.join(exp_dir, "output.nii"))

        print(f"Processing Done. Dataset is saved to {exp_dir}.")

    else:
        raise ValueError(
            "Invalid input: Provide a .txt file for a trained experiment or a directory containing raw data."
        )


if __name__ == "__main__":
    main()
