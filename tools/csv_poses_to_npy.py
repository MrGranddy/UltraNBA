import argparse

import numpy as np

from utils.pose import convert_to_homogeneous, load_pose_data


def main(args):

    poses = np.array(load_pose_data(args.input))
    poses = convert_to_homogeneous(poses)

    np.save(args.output, poses)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Converts poses in CSV format to NPY format."
    )
    parser.add_argument(
        "input", type=str, help="Input tracking file, can be '.npy', '.csv' or '.ts'."
    )
    parser.add_argument(
        "output", type=str, help="The output tracking file, can only be '.npy'."
    )

    args = parser.parse_args()

    main(args)
