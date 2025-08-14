import os

import numpy as np
import torch

from camera import Lie, Pose


def convert_to_closest_valid_transformation(T):
    """Some of the original transformations in the dataset are not valid due to numerical errors. We convert them to the closest valid transformation.

    Args:
        T (np.ndarray): A batch of 3x4 transformation matrices with shape (B, 3, 4) or (3, 4).

    Returns:
        np.ndarray: A batch of 3x4 transformation matrices.
    """

    if len(T.shape) == 2:
        T = T[None, ...]

    R = T[:, :3, :3]
    t = T[:, :3, 3:4]

    # Convert the invalid transformation to the closest valid transformation
    U, S, V = np.linalg.svd(R)
    R = U @ V

    # Check if the determinant is negative
    det = np.linalg.det(R)
    U[det < 0, :, -1] *= -1

    R = U @ V

    return np.concatenate([R, t], axis=-1)


def create_transformation(
    x_angle=0.0, y_angle=0.0, z_angle=0.0, tx=0.0, ty=0.0, tz=0.0
):
    """
        Creates a transformation matrix with the given parameters.

    Args:
        x_angle (float): Rotation around the X-axis in degrees.
        y_angle (float): Rotation around the Y-axis in degrees.
        z_angle (float): Rotation around the Z-axis in degrees.
        tx (float): Translation along the X-axis.
        ty (float): Translation along the Y-axis.
        tz (float): Translation along the Z-axis.

    Returns:
        np.ndarray: Transformation matrix.
    """

    # Convert degrees to radians
    x_rad, y_rad, z_rad = np.radians(x_angle), np.radians(y_angle), np.radians(z_angle)

    # Rotation matrix around X-axis
    Rx = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(x_rad), -np.sin(x_rad), 0],
            [0, np.sin(x_rad), np.cos(x_rad), 0],
            [0, 0, 0, 1],
        ]
    )

    # Rotation matrix around Y-axis
    Ry = np.array(
        [
            [np.cos(y_rad), 0, np.sin(y_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(y_rad), 0, np.cos(y_rad), 0],
            [0, 0, 0, 1],
        ]
    )

    # Rotation matrix around Z-axis
    Rz = np.array(
        [
            [np.cos(z_rad), -np.sin(z_rad), 0, 0],
            [np.sin(z_rad), np.cos(z_rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # Combined Rotation Matrix: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx

    # Translation matrix
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]  # Apply translation

    # Final transformation (Camera to World)
    camera_to_world = T @ R  # First rotate, then translate

    return camera_to_world


def apply_global_registration(
    poses: np.ndarray, x_angle=0.0, y_angle=0.0, z_angle=0.0, tx=0.0, ty=0.0, tz=0.0
) -> np.ndarray:
    """
    Applies a global registration transformation to a set of Nx4x4 matrices
    using specified rotation angles and translation values.

    Args:
        poses (np.ndarray): An Nx4x4 array representing transformation matrices.
        x_angle (float): Rotation around the X-axis in degrees.
        y_angle (float): Rotation around the Y-axis in degrees.
        z_angle (float): Rotation around the Z-axis in degrees.
        tx (float): Translation along the X-axis.
        ty (float): Translation along the Y-axis.
        tz (float): Translation along the Z-axis.

    Returns:
        np.ndarray: Transformed Nx4x4 matrices.
    """
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("Poses must be an (N, 4, 4) array")

    camera_to_world = create_transformation(x_angle, y_angle, z_angle, tx, ty, tz)

    # Apply transformation to all pose matrices
    transformed_poses = np.array([camera_to_world @ pose for pose in poses])

    return transformed_poses


def apply_calibration(
    poses: np.ndarray, x_angle=0.0, y_angle=0.0, z_angle=0.0, tx=0.0, ty=0.0, tz=0.0
) -> np.ndarray:
    """
    Applies a calibration transformation to a set of Nx4x4 matrices
    using specified rotation angles and translation values.

    Args:
        poses (np.ndarray): An Nx4x4 array representing transformation matrices.
        x_angle (float): Rotation around the X-axis in degrees.
        y_angle (float): Rotation around the Y-axis in degrees.
        z_angle (float): Rotation around the Z-axis in degrees.
        tx (float): Translation along the X-axis.
        ty (float): Translation along the Y-axis.
        tz (float): Translation along the Z-axis.

    Returns:
        np.ndarray: Transformed Nx4x4 matrices.
    """
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("Poses must be an (N, 4, 4) array")

    camera_to_world = create_transformation(x_angle, y_angle, z_angle, tx, ty, tz)

    # Apply transformation to all pose matrices
    transformed_poses = np.array([pose @ camera_to_world for pose in poses])

    return transformed_poses


def save_to_csv(filename: str, data: np.ndarray) -> None:
    """
    Save a 2D numpy array to a CSV file.

    Args:
        filename (str): The output file name.
        data (np.ndarray): The data to be saved.
    """
    np.savetxt(filename, data, delimiter=",", fmt="%s")


def load_pose_data(pose_path: str, make_valid=False) -> torch.Tensor:
    """
    Load and optionally validate a set of camera poses.

    Args:
        pose_path (str): Path to the file containing poses.

    Returns:
        torch.Tensor: Loaded pose data (N, 3, 4).
    """
    if pose_path.endswith(".npy"):
        poses = np.load(pose_path)[:, :3, :4]

    elif pose_path.endswith(".csv"):
        t = []
        with open(pose_path, "r") as f:
            for line in f:
                t.append(list(map(float, line.strip().split(","))))

        poses = np.array(t).reshape(len(t), 4, 4).transpose(0, 2, 1)[:, :3, :4]

    elif pose_path.endswith(".ts"):
        t = []
        with open(pose_path, "r") as f:
            for line in f:
                t.append(list(map(float, line.strip().split()))[:16])
        poses = np.array(t).reshape(len(t), 4, 4).transpose(0, 2, 1)[:, :3, :4]

    else:
        raise ValueError("Unsupported file format")

    if make_valid:
        raise NotImplementedError(
            "We do no advice using this option, if you still want to use it, use it at your own risk."
        )
        # poses = convert_to_closest_valid_transformation(poses)

    return torch.tensor(poses, dtype=torch.float32)


def flatten_poses(poses: np.ndarray) -> np.ndarray:
    """
    Flatten (N, 4, 4) camera poses into a (N, 16) format for CSV saving.

    Args:
        poses (np.ndarray): Homogeneous camera pose matrices.

    Returns:
        np.ndarray: Flattened camera poses.
    """
    return poses.transpose(0, 2, 1).reshape(poses.shape[0], -1)


def convert_to_homogeneous(poses: np.ndarray) -> np.ndarray:
    """
    Convert a set of 3x4 camera pose matrices to 4x4 homogeneous form.

    Args:
        poses (np.ndarray): Input (N, 3, 4) camera poses.

    Returns:
        np.ndarray: Output (N, 4, 4) homogeneous camera poses.
    """
    return np.concatenate(
        [poses, np.tile([[0, 0, 0, 1]], (poses.shape[0], 1, 1))], axis=1
    )


def reflect_image_poses(poses: np.ndarray, axis: str) -> np.ndarray:
    """
    Reflects a batch of 4x4 pose matrices in place, keeping their global positions fixed
    while changing their orientations as if the images are being reflected across a plane.

    Parameters:
        poses (np.ndarray): An (N, 4, 4) array of pose matrices.
        axis (str): The axis to reflect across ('x', 'y', or 'z').

    Returns:
        np.ndarray: The updated (N, 4, 4) array with reflection applied in local space.
    """
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("Poses must be an (N, 4, 4) array")

    # Define reflection matrices (applied in local coordinates)
    if axis.lower() == "x":
        R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif axis.lower() == "y":
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    elif axis.lower() == "z":
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    # Apply reflection to all poses
    new_poses = poses.copy()
    new_poses[:, :3, :3] = poses[:, :3, :3] @ R  # Reflect orientation
    return new_poses


### UTILITY CLASSES AND FUNCTIONS FOR ANALYSIS ###


class PoseLoader:
    """Class to handle loading and refining poses for given checkpoints."""

    def __init__(self, pose_path, data_dir, weights_dir):
        self.pose_path = pose_path
        self.data_dir = data_dir
        self.weights_dir = weights_dir

    @staticmethod
    def preprocess_poses(poses):
        """Preprocess poses by scaling and flipping y-axis."""
        poses[:, :3, 3] *= 0.001
        flip_y = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        poses = np.stack([pose @ flip_y for pose in poses])
        return poses

    def load_original_and_noisy_poses(self):
        """Load and preprocess noisy and original poses."""
        noisy_poses = load_pose_data(self.pose_path).cpu().numpy()
        original_poses = (
            load_pose_data(os.path.join(self.data_dir, "poses.npy")).cpu().numpy()
        )

        noisy_poses = self.preprocess_poses(noisy_poses)
        original_poses = self.preprocess_poses(original_poses)

        return noisy_poses, original_poses

    def load_weights(self, iteration):
        """Load checkpoint weights for given iteration."""
        weights_path = os.path.join(self.weights_dir, f"{iteration:06d}.tar")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
        return ckpt["pose_refine_state_dict"]["se3_refine.weight"].cpu()

    def refine_poses(self, noisy_poses, pose_refine_weights):
        """Refine noisy poses using specified refinement weights."""
        lie = Lie()
        pose_util = Pose()

        torch_noisy_poses = torch.tensor(noisy_poses).float()
        refined_poses = torch.zeros_like(torch_noisy_poses)
        for i in range(torch_noisy_poses.shape[0]):
            refine_mat = lie.se3_to_SE3(pose_refine_weights[i])
            refine_mat[:, 3] *= 0.001
            refined_poses[i] = pose_util.compose([refine_mat, torch_noisy_poses[i]])

        return refined_poses.cpu().numpy()

    def process_iterations(self, iterations):
        """
        Refine poses for multiple iterations.

        Args:
            iterations (list): List of iteration numbers.

        Returns:
            dict: Dictionary of refined poses indexed by iteration numbers.
        """
        noisy_poses, original_poses = self.load_original_and_noisy_poses()

        refined_poses_dict = {}

        for iteration in iterations:
            pose_refine_weights = self.load_weights(iteration)
            refined_poses = self.refine_poses(noisy_poses, pose_refine_weights)
            refined_poses_dict[iteration] = refined_poses

        return {
            "original": original_poses,
            "noisy": noisy_poses,
            "refined": refined_poses_dict,
        }


# Example boilerplate function for notebook usage
def load_and_refine_poses(data_dir, pose_path, weights_dir, iterations):
    loader = PoseLoader(pose_path, data_dir, weights_dir)
    poses_data = loader.process_iterations(iterations)
    return poses_data
