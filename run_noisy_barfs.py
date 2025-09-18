import argparse
import os
import subprocess

tss = [0.15, 0.3]
rss = [0.07, 0.15]
perturb_ratios = [1.0]
repeats = 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run noisy BARF experiments")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="UltraNeRF dataset directory"
    )
    parser.add_argument(
        "--base_config_barf", type=str, help="Base Config for the experiment (BARF)."
    )
    parser.add_argument(
        "--base_config_nerf", type=str, help="Base Config for the experiment (NeRF)."
    )
    parser.add_argument(
        "--identifier",
        type=str,
        default="misc",
        help="Simple identifier to create expname.",
    )
    args = parser.parse_args()

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

            expname = f"{args.identifier}_barf_reg_rot_{rotation_strength}_tr_{translation_strength}_pr_{pr}_{i}"
            pose_path = os.path.join(
                args.data_dir,
                "noisy_poses",
                f"{rotation_strength}_{translation_strength}_{pr}_{i}.npy",
            )

            # Construct the training command
            train_command = [
                "python",
                "run_ultranba.py",
                "--expname",
                expname,
                "--pose_path",
                pose_path,
                "--config",
                args.base_config_barf,
                "--tensorboard",
                # "--n_iters", "50",
                # "--i_print", "10",
                # "--i_weights", "10",
                "--i_weights",
                "20000",
                "--reg",
            ]

            # Print and execute the command
            print(f"Running command for BARF: {' '.join(train_command)}")
            try:
                subprocess.run(train_command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Command failed with return code {e.returncode}")

            expname = f"{args.identifier}_nerf_reg_rot_{rotation_strength}_tr_{translation_strength}_pr_{pr}_{i}"

            train_command = [
                "python",
                "run_ultranerf.py",
                "--expname",
                expname,
                "--pose_path",
                pose_path,
                "--config",
                args.base_config_nerf,
                "--tensorboard",
                # "--n_iters", "50",
                # "--i_print", "10",
                # "--i_weights", "10",
                "--i_weights",
                "20000",
                "--reg",
            ]

            # Print and execute the command
            print(f"Running command for Ultranerf: {' '.join(train_command)}")
            try:
                subprocess.run(train_command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Command failed with return code {e.returncode}")
