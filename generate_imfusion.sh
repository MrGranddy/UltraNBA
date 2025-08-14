#!/bin/bash

# Define the lists for variations
LIVER_VARIANTS=("liver_long" "lab_al_long" "lab_alx2_long" "wan_ik_long" "wan_ikx2_long")  # Replace with your actual list
ROTATIONS=("0.07" "0.15")  # Replace with your actual list
TRANSLATIONS=("0.15" "0.3")  # Replace with your actual list

# Iterate over all combinations
for liver in "${LIVER_VARIANTS[@]}"; do
    for rot in "${ROTATIONS[@]}"; do
        for tr in "${TRANSLATIONS[@]}"; do
            log_dir="./logs/${liver}_barf_reg_rot_${rot}_tr_${tr}_pr_1.0_0"
            args_file="${log_dir}/args.txt"

            # Check if args.txt exists before running
            if [[ -f "$args_file" ]]; then
                echo "Running experiment for: ${liver}, rot=${rot}, tr=${tr}"
                python -m tools.experiment_to_imfusion "$args_file"
            else
                echo "Skipping: $args_file not found!"
            fi
        done
    done
done
