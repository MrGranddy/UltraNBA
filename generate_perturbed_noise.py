import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from camera import Lie, Pose
from utils.test import rotation_translation_errors
from utils.vis import visualize_poses

# Set noisy position config
tss = [0.15, 0.3]
rss = [0.07, 0.15]
perturb_ratios = [1.0]
repeats = 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate perturbed noise")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="UltraNeRF dataset directory"
    )

    args = parser.parse_args()

    lie = Lie()
    pose = Pose()

    data_dir = args.data_dir

    # Create configs for each combination except ts and rs both 0.0
    configs = []
    for ts in tss:
        for rs in rss:
            if ts == 0.0 and rs == 0.0:
                continue
            for pr in perturb_ratios:
                for _ in range(repeats):
                    configs.append((ts, rs, pr))

    for ts, rs, pr in configs:
        rotation_strength = rs
        translation_strength = ts

        print(
            f"Rotation Strength: {rotation_strength}, Translation Strength: {translation_strength}, Perturb Ratio: {pr}"
        )

        for i in range(repeats):

            # Load poses
            poses_path = os.path.join(data_dir, "poses.npy")
            poses_np = np.load(poses_path)[:, :3, :4]

            # Load Images (for visualization only)
            image_path = os.path.join(data_dir, "images.npy")
            image = np.load(image_path)
            _, H, W = image.shape

            # Convert to tensor
            poses = torch.tensor(poses_np, dtype=torch.float32)

            num_poses = poses.shape[0]

            # Generate noise
            se3_noise = torch.randn(num_poses, 6)
            se3_noise[:, :3] *= rotation_strength
            se3_noise[:, 3:] *= translation_strength

            # Convert SE(3) noise and apply it
            SE3_noise = lie.se3_to_SE3(se3_noise)
            noisy_poses = pose.compose([SE3_noise, poses])

            # Convert to numpy for visualization
            org_poses = poses.numpy()
            noisy_poses = noisy_poses.numpy()

            # Create ID
            id_pose = f"{rotation_strength}_{translation_strength}_{pr}_{i}"

            # Visualize the original and perturbed poses
            visualize_poses(
                org_poses,
                noisy_poses,
                sample_ratio=0.1,
                arrow_length=5.0,
                title=id_pose,
            )
            vis_path = os.path.join(data_dir, "noisy_vis")
            os.makedirs(vis_path, exist_ok=True)
            plt.savefig(vis_path)
            plt.close()

            # Calculate rotation and translation errors
            errors = rotation_translation_errors(org_poses, noisy_poses)
            print(
                f"Translation: {errors['translation_mean']:.4f} ± {errors['translation_std']:.4f} (mm), "
                f"Rotation: {errors['rotation_mean']:.4f} ± {errors['rotation_std']:.4f} (degrees)"
            )
            # Report translation scale of original poses in X and Y directions
            print(
                f"Translation scale in X direction: {np.min(org_poses[:, 0, 3])} - {np.max(org_poses[:, 0, 3])}, Differece: {np.max(org_poses[:, 0, 3]) - np.min(org_poses[:, 0, 3])}"
            )
            print(
                f"Translation scale in Y direction: {np.min(org_poses[:, 1, 3])} - {np.max(org_poses[:, 1, 3])}, Differece: {np.max(org_poses[:, 1, 3]) - np.min(org_poses[:, 1, 3])}"
            )

            # Select slices to apply the noise
            num_slices = int(pr * poses.shape[0])
            selected = np.random.permutation(poses.shape[0])[:num_slices]

            org_poses[selected] = noisy_poses[selected]
            noisy_poses = org_poses

            # Convert back to 4x4 format

            noisy_poses = np.concatenate(
                [
                    noisy_poses,
                    np.repeat(
                        np.array([0, 0, 0, 1]).reshape(1, 1, 4),
                        repeats=noisy_poses.shape[0],
                        axis=0,
                    ),
                ],
                axis=1,
            )

            # Save noisy poses
            os.makedirs(os.path.join(data_dir, "noisy_poses"), exist_ok=True)
            noisy_poses_path = os.path.join(data_dir, "noisy_poses", f"{id_pose}.npy")
            np.save(noisy_poses_path, noisy_poses)
