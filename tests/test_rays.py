import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch

from utils.nerf import get_rays_us_linear


def visualize_full_volume_pyvista(
    H, W, sh, sw, poses, images, ratio=1.0, downsample_factor=0.5, ray_ratio=0.05
):
    """
    Visualizes 2D ultrasound slices in 3D space using PyVista along with sampled rays.

    Args:
        H (int): Height of the ultrasound images.
        W (int): Width of the ultrasound images.
        sh (float): Scaling factor for height.
        sw (float): Scaling factor for width.
        poses (np.ndarray): Array of 4x4 transformation matrices for each image.
        images (list): List of ultrasound images corresponding to the poses.
        rays_o (np.ndarray): Ray origins of shape (N, B, 3).
        rays_d (np.ndarray): Ray directions of shape (N, B, 3).
        ratio (float, optional): Fraction of poses to visualize (default: 1.0, full set).
        downsample_factor (float, optional): Factor to downsample images for performance (default: 0.5).
        ray_ratio (float, optional): Fraction of rays per image to visualize (default: 0.05).
    """
    rays_o, rays_d = [], []
    for pose in poses:
        o, d = get_rays_us_linear(H, W, sh, sw, torch.tensor(pose).float())
        rays_o.append(o.numpy())
        rays_d.append(d.numpy())

    rays_o = np.stack(rays_o, axis=0)
    rays_d = np.stack(rays_d, axis=0)

    assert (
        len(poses) == len(images) == len(rays_o) == len(rays_d)
    ), "Mismatch: poses, images, and rays must have the same length."

    # Initialize PyVista plotter
    plotter = pv.Plotter()

    # Select a subset of poses and images based on ratio
    num_selected = int(ratio * len(poses))
    perm = np.random.permutation(len(poses))[:num_selected]
    poses, images, rays_o, rays_d = (
        poses[perm],
        images[perm],
        rays_o[perm],
        rays_d[perm],
    )

    # Process each image and its corresponding pose
    for i, pose in enumerate(poses):
        img = images[i]

        # Downsample image for performance
        if downsample_factor != 1.0:
            img = cv2.resize(
                img, (int(W * downsample_factor), int(H * downsample_factor))
            )

        # Convert grayscale images to RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)

        # Set alpha to 0 for pixels where the sum of RGB values is below a threshold
        threshold = 15  # Adjust as needed
        img[np.sum(img[..., :3], axis=-1) < threshold, 3] = 0

        # Define the image plane in PyVista
        plane = pv.Plane(i_size=W * sw, j_size=H * sh, i_resolution=1, j_resolution=1)
        plane.transform(pose)

        # Create and assign texture
        texture = pv.Texture(img)
        plotter.add_mesh(plane, texture=texture, show_edges=False)

        # Select subset of rays based on ray_ratio
        num_selected_rays = int(ray_ratio * rays_o.shape[1])
        selected_indices = np.random.choice(
            rays_o.shape[1], num_selected_rays, replace=False
        )

        for j in selected_indices:
            origin = rays_o[i, j]
            direction = rays_d[i, j] / np.linalg.norm(
                rays_d[i, j]
            )  # Normalize direction
            end = origin + direction * (W * sw)  # Extend the rays for visibility

            # Add ray as a line in PyVista
            plotter.add_mesh(pv.Line(origin, end), color="blue", line_width=1)
            # Add a small red sphere at the start of the ray
            plotter.add_mesh(
                pv.Sphere(radius=W * sw * 0.01, center=origin), color="red"
            )

    # Configure visualization settings
    plotter.set_background("white")

    # Add axis labels and grid
    plotter.show_bounds(
        grid=True,
        xtitle="X (m)",
        ytitle="Y (m)",
        ztitle="Z (m)",
        location="outer",
        ticks="both",
    )

    # Show the 3D visualization
    plotter.show()


if __name__ == "__main__":

    data_name = "WAN_IKx2"

    # Load poses
    pose_path = f"G:/ultrasound_data/original/{data_name}/poses.npy"
    image_path = f"G:/ultrasound_data/original/{data_name}/images.npy"

    poses = np.load(pose_path)
    images = np.load(image_path)
    images = images.transpose(0, 2, 1)

    poses[:, :3, 3] *= 0.001
    flip_y = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    poses = np.stack([pose @ flip_y for pose in poses])

    pd = 93
    pw = 38

    # Example parameters
    scaling = 0.001
    near = 0
    probe_depth = pd * scaling
    probe_width = pw * scaling
    far = probe_depth

    H, W = images.shape[1], images.shape[2]
    sw = probe_depth / float(W)
    sh = probe_width / float(H)

    # visualize_full_volume(H, W, sw, sh, poses, ratio=0.05)
    visualize_full_volume_pyvista(
        H, W, sh, sw, poses, images, ratio=0.05, ray_ratio=0.05
    )
