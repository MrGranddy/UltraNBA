#!/bin/sh
 
#SBATCH --job-name=train_perturbed_lab_al
#SBATCH --output=slurm/logs/train_perturbed_lab_al.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=slurm/logs/train_perturbed_lab_al.err  # Standard error of the script
#SBATCH --account=phds
#SBATCH --partition=phds
#SBATCH --time=2-00:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=12  # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=48G  # Memory in GB (Don't use more than 48/24 per GPU unless you absolutely need it and know what you are doing)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ultrabarf
python run_noisy_barfs.py \
    --data_dir /home/guests/{NAME}/ultrasound_data/original/LAB_AL \
    --base_config_barf configs/config_base_barf_lab_al.txt \
    --base_config_nerf configs/config_base_nerf_lab_al.txt --identifier "LAB_AL"