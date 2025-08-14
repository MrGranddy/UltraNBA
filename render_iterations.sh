#!/bin/bash

# Define datasets, rotation values, and translation values
datasets=("lab_alx2" "wan_ik" "wan_ikx2" "lab_al" "liver")
rotations=("0.07" "0.15")
translations=("0.15" "0.3")

# Iterate over all combinations and execute the command
for dataset in "${datasets[@]}"; do
    for rot in "${rotations[@]}"; do
        for tr in "${translations[@]}"; do
            args_path="./logs/${dataset}_long_nerf_reg_rot_${rot}_tr_${tr}_pr_1.0_0/args.txt"
            if [[ -f "$args_path" ]]; then
                echo "Running: python -m render_us $args_path"
                python -m render_us "$args_path"
            else
                echo "Skipping: $args_path (file not found)"
            fi
        done
    done
done
