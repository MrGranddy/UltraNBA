import argparse
import json

import numpy as np

from utils.pose import (
    apply_calibration,
    apply_global_registration,
    convert_to_homogeneous,
    flatten_poses,
    load_pose_data,
    reflect_image_poses,
    save_to_csv,
)


def main(args):

    poses = load_pose_data(args.input)
    poses = np.array(poses)
    poses = convert_to_homogeneous(poses)

    if args.calibration:
        with open(args.calibration, "r") as f:
            params = json.load(f)
            poses = apply_calibration(poses, **params)

    if args.registration:
        with open(args.registration, "r") as f:
            params = json.load(f)
            poses = apply_global_registration(poses, **params)

    if args.options:
        with open(args.options, "r") as f:
            params = json.load(f)

        if params["vertical_flip"]:
            poses = reflect_image_poses(poses, axis="x")
        if params["horizontal_flip"]:
            poses = reflect_image_poses(poses, axis="y")

    flat_poses = flatten_poses(poses)
    save_to_csv(args.output, flat_poses)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Applies calibration and/or registration transforms to a given tracking."
    )
    parser.add_argument(
        "input", type=str, help="Input tracking file, can be '.npy', '.csv' or '.ts'."
    )
    parser.add_argument(
        "output", type=str, help="The output tracking file, can only be '.csv'."
    )
    parser.add_argument(
        "--calibration",
        type=str,
        help="Path to JSON file storing calibration parameters.",
    )
    parser.add_argument(
        "--registration",
        type=str,
        help="Path to JSON file storing registration parameters.",
    )
    parser.add_argument(
        "--options", type=str, help="Path to JSON file storing additional options."
    )

    args = parser.parse_args()

    main(args)
