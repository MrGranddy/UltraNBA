import argparse
import os
import shutil

import numpy as np
import torch

from load_us import load_us_data
from utils.nerf import create_barf, create_nerf, render_us

# params_to_save = ["intensity_map", "attenuation_total", "b", "confidence_maps", "attenuation_coeff", "r", "reflection_coeff", "reflection_total", "r_amplified", "scatter_amplitude"]
params_to_save = ["intensity_map"]


def parse_experiment_args(args_path):
    args = {}
    with open(args_path, "r") as f:
        for line in f.readlines():
            key, value = line.strip().split(" = ")
            try:
                args[key] = int(value)
            except ValueError:
                try:
                    args[key] = float(value)
                except ValueError:
                    args[key] = None if value == "None" else value
    return argparse.Namespace(**args)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser_console = argparse.ArgumentParser()
    parser_console.add_argument("args", type=str, default="logs")
    parser_console.add_argument("--model_no", type=str, default=None)
    args_console = parser_console.parse_args()

    args = parse_experiment_args(args_console.args)

    if "barf" in args.expname:
        method = "barf"
    else:
        method = "nerf"

    if args_console.model_no:
        model_no = args_console.model_no
    else:
        model_no = args.n_iters

    model_no = "{:06d}".format(int(model_no))

    weights_path = os.path.join(args.basedir, args.expname, f"{model_no}.tar")
    weights = torch.load(weights_path)

    pose_path = args.pose_path.replace("/home/guests/{NAME}", "G:").replace(
        "/home/guests/{NAME}", "G:"
    )
    datadir = args.datadir.replace("/home/guests/{NAME}", "G:").replace(
        "/home/guests/{NAME}", "G:"
    )

    data = load_us_data(datadir, pose_path=pose_path)

    images = data["images"]
    poses = data["poses"]
    i_test = data["i_test"]
    if pose_path:
        org_poses = data["org_poses"]
        org_poses = org_poses[:, :3, :4]

    scaling = 0.001
    near = 0
    probe_depth = args.probe_depth * scaling
    probe_width = args.probe_width * scaling
    far = probe_depth

    images = images.transpose(0, 2, 1)

    H, W = images.shape[1], images.shape[2]
    sw = probe_depth / float(W)
    sh = probe_width / float(H)

    # Create nerf model
    if method == "nerf":
        model_ret = create_nerf(args, device=device, mode="test")
    elif method == "barf":
        model_ret = create_barf(
            torch.tensor(poses[:, :3, :4]).float(), args, device, mode="test"
        )

    render_kwargs_test = model_ret[1]

    if method == "nerf":
        print(f"Loading weights into NeRF model from {weights_path}")
        render_kwargs_test["network_fn"].load_state_dict(
            weights["network_fn_state_dict"]
        )

    elif method == "barf":
        print(f"Loading weights into BARF model from {weights_path}")
        render_kwargs_test["network_fn"].load_state_dict(
            weights["network_fn_state_dict"]
        )
        render_kwargs_test["pose_refine"].load_state_dict(
            weights["pose_refine_state_dict"]
        )

    # Define output dir for renders
    main_render_out_dir = "render_outputs"
    os.makedirs(main_render_out_dir, exist_ok=True)

    output_dir = os.path.join(main_render_out_dir, f"{args.expname}_{model_no}")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    params = {}

    # Convert poses and run rendering
    with torch.no_grad():
        for i, c2w in enumerate(poses):

            if method == "nerf":
                c2w_torch = torch.from_numpy(c2w[:3, :4]).to(device).unsqueeze(0)
            elif method == "barf":
                c2w_torch = (
                    render_kwargs_test["pose_refine"].get_pose(int(i)).unsqueeze(0)
                )

            # render_us returns a dict of torch tensors
            rendering_output = render_us(
                H,
                W,
                sh,
                sw,
                c2w=c2w_torch,
                chunk=args.chunk,
                retraw=True,
                near=near,
                far=far,
                **render_kwargs_test,
            )

            for param, map in rendering_output.items():
                if param not in params:
                    params[param] = np.zeros(
                        (poses.shape[0], map.shape[0], map.shape[1]), dtype="float32"
                    )
                params[param][i] = map.detach().cpu().numpy()

            if (i + 1) % 100 == 0:
                print(f"{(i+1)}/{poses.shape[0]} done.")

    for param, map in params.items():

        # Compute percentiles to clip extreme outliers
        lower_bound = np.percentile(map, 1)  # Lower 1 percentile
        upper_bound = np.percentile(map, 99)  # Upper 99 percentile

        # Clip outliers
        map = np.clip(map, lower_bound, upper_bound)

        # Normalize to 0-255
        map = (map - map.min()) / (map.max() - map.min() + 1.0e-6) * 255

        # Convert to uint8
        map = map.astype(np.uint8)

        # Transpose H and W
        map = map.transpose(0, 2, 1)

        # Save the processed map
        np.save(os.path.join(output_dir, f"{param}.npy"), map)
