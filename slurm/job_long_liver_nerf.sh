#!/bin/sh
 
#SBATCH --job-name=train_long_liver
#SBATCH --output=slurm/logs/train_long_liver.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=slurm/logs/train_long_liver.err  # Standard error of the script
#SBATCH --account=students
#SBATCH --partition=students
#SBATCH --time=1-00:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=12  # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=48G  # Memory in GB (Don't use more than 48/24 per GPU unless you absolutely need it and know what you are doing)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ultrabarf

# Define arrays for parameters
tss=(0.15 0.3)
rss=(0.07 0.15)
perturb_ratios=(1.0)
repeats=1

base_config_nerf="configs/config_long_barf_liver.txt"
identifier="liver_long"
data_dir="/home/guests/{NAME}/ultrasound_data/original/liver"

# Iterate over parameter combinations
for ts in "${tss[@]}"; do
    for rs in "${rss[@]}"; do
        if [[ "$ts" == "0.0" && "$rs" == "0.0" ]]; then
            continue
        fi
        for pr in "${perturb_ratios[@]}"; do
            for ((i=0; i<$repeats; i++)); do
                
                rotation_strength="$rs"
                translation_strength="$ts"
                
                echo "Rotation Strength: $rotation_strength, Translation Strength: $translation_strength, Perturb Ratio: $pr"
                
                expname="${identifier}_nerf_reg_rot_${rotation_strength}_tr_${translation_strength}_pr_${pr}_${i}"
                pose_path="${data_dir}/noisy_poses/${rotation_strength}_${translation_strength}_${pr}_${i}.npy"
                
                # Construct the training command
                train_command=(
                    "python" "run_ultranerf.py"
                    "--expname" "$expname"
                    "--pose_path" "$pose_path"
                    "--config" "$base_config_nerf"
                    "--tensorboard"
                    "--i_weights" "10000"
                    "--reg"
                    "--multires" "16"
                )
                
                # Print and execute the command
                echo "Running command for NeRF: ${train_command[*]}"
                "${train_command[@]}"
                if [[ $? -ne 0 ]]; then
                    echo "Command failed with return code $?"
                fi
            done
        done
    done
done
