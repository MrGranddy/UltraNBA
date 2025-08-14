import io

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def visualize_poses(
    original_poses, perturbed_poses, sample_ratio=1.0, arrow_length=0.5, title=None
):
    """
    Visualizes sampled original and perturbed camera poses in 3D space with reoriented axes and uniform axis scaling.
    In this orientation:
        - X is the width (image horizontal axis).
        - Y is the height (image vertical axis).
        - Z corresponds to the slices or depth.

    Parameters:
        original_poses (np.ndarray): Array of shape (N, 3, 4), original poses.
        perturbed_poses (np.ndarray): Array of shape (N, 3, 4), perturbed poses.
        sample_ratio (float): Ratio of samples to visualize.
        arrow_length (float): Length of the arrows representing camera directions.
    """

    # Determine the number of samples
    num_samples = int(original_poses.shape[0] * sample_ratio)
    indices = np.linspace(0, original_poses.shape[0] - 1, num_samples, dtype=int)

    # Sample the poses
    sampled_original_poses = original_poses[indices]
    sampled_perturbed_poses = perturbed_poses[indices]

    # Extract translations and directions
    original_translations = sampled_original_poses[:, :, 3]
    perturbed_translations = sampled_perturbed_poses[:, :, 3]
    original_directions = sampled_original_poses[:, :, 2]  # Forward direction (z-axis)
    perturbed_directions = sampled_perturbed_poses[:, :, 2]

    # Reorient axes
    original_reoriented = original_translations[
        :, [0, 2, 1]
    ]  # Map X, Y, Z to width, slices, height
    perturbed_reoriented = perturbed_translations[:, [0, 2, 1]]
    original_directions_reoriented = original_directions[:, [0, 2, 1]]
    perturbed_directions_reoriented = perturbed_directions[:, [0, 2, 1]]

    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot translations
    ax.scatter(
        original_reoriented[:, 0],
        original_reoriented[:, 1],
        original_reoriented[:, 2],
        color="blue",
        label="Original Translations",
        s=50,
        depthshade=True,
    )
    ax.scatter(
        perturbed_reoriented[:, 0],
        perturbed_reoriented[:, 1],
        perturbed_reoriented[:, 2],
        color="red",
        label="Perturbed Translations",
        s=50,
        depthshade=True,
    )

    # Plot directions as quivers
    for i in range(num_samples):
        ax.quiver(
            original_reoriented[i, 0],
            original_reoriented[i, 1],
            original_reoriented[i, 2],
            original_directions_reoriented[i, 0],
            original_directions_reoriented[i, 1],
            original_directions_reoriented[i, 2],
            color="blue",
            length=arrow_length,
            normalize=True,
            alpha=0.7,
        )
        ax.quiver(
            perturbed_reoriented[i, 0],
            perturbed_reoriented[i, 1],
            perturbed_reoriented[i, 2],
            perturbed_directions_reoriented[i, 0],
            perturbed_directions_reoriented[i, 1],
            perturbed_directions_reoriented[i, 2],
            color="red",
            length=arrow_length,
            normalize=True,
            alpha=0.7,
        )

    # Set labels and legend
    ax.set_xlabel("Width (X)")
    ax.set_ylabel("Slices (Z)")
    ax.set_zlabel("Height (Y)")
    # ax.set_title(f"Reoriented Visualization of Sampled Poses (Sample Ratio: {sample_ratio})")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True)

    # Uniform aspect ratio for all axes
    all_points = np.concatenate([original_reoriented, perturbed_reoriented], axis=0)
    max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 2.0
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) / 2.0
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) / 2.0
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def show_colorbar(image, cmap="rainbow"):
    figure = plt.figure(figsize=(5, 5))
    plt.imshow(image.numpy(), cmap=cmap)
    plt.colorbar()
    buf = io.BytesIO()
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format="png")
    plt.close(figure)
    return buf


def define_image_grid_3D_np(x_size, y_size):
    y = np.array(range(x_size))
    x = np.array(range(y_size))
    xv, yv = np.meshgrid(x, y, indexing="ij")
    image_grid_xy = np.vstack((xv.ravel(), yv.ravel()))
    z = np.zeros(image_grid_xy.shape[1])
    image_grid = np.vstack((image_grid_xy, z))
    return image_grid


def plot_pose(ax: Axes3D, pose: np.ndarray, color: str, label: str) -> None:
    """
    Plot a camera pose as a coordinate frame in 3D.

    Args:
        ax (Axes3D): Matplotlib 3D axis.
        pose (np.ndarray): 3x4 pose matrix.
        color (str): Color for the origin point.
        label (str): Label for the pose.
    """
    origin = pose[:, 3]
    x_axis = origin + pose[:, 0] * 0.05
    y_axis = origin + pose[:, 1] * 0.05
    z_axis = origin + pose[:, 2] * 0.05

    ax.quiver(*origin, *(x_axis - origin), color="r", length=5)
    ax.quiver(*origin, *(y_axis - origin), color="g", length=5)
    ax.quiver(*origin, *(z_axis - origin), color="b", length=5)
    ax.scatter(*origin, color=color, label=label, s=10)
