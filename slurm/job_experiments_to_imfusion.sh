#!/bin/sh
 
#SBATCH --job-name=experiments_to_imfusion
#SBATCH --output=slurm/logs/experiments_to_imfusion.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=slurm/logs/experiments_to_imfusion.err  # Standard error of the script
#SBATCH --account=students
#SBATCH --partition=students
#SBATCH --time=0-12:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:0  # Number of GPUs if needed
#SBATCH --cpus-per-task=2  # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=32G  # Memory in GB (Don't use more than 48/24 per GPU unless you absolutely need it and know what you are doing)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ultrabarf

python -m tools.experiment_to_imfusion logs/liver_barf_reg_rot_0.07_tr_0.15_pr_1.0_0/args.txt
python -m tools.experiment_to_imfusion logs/liver_barf_reg_rot_0.07_tr_0.3_pr_1.0_0/args.txt
python -m tools.experiment_to_imfusion logs/liver_barf_reg_rot_0.15_tr_0.15_pr_1.0_0/args.txt
python -m tools.experiment_to_imfusion logs/liver_barf_reg_rot_0.15_tr_0.3_pr_1.0_0/args.txt
python -m tools.experiment_to_imfusion logs/LAB_AL_barf_reg_rot_0.07_tr_0.15_pr_1.0_0/args.txt
python -m tools.experiment_to_imfusion logs/LAB_AL_barf_reg_rot_0.07_tr_0.3_pr_1.0_0/args.txt
python -m tools.experiment_to_imfusion logs/LAB_AL_barf_reg_rot_0.15_tr_0.15_pr_1.0_0/args.txt
python -m tools.experiment_to_imfusion logs/LAB_AL_barf_reg_rot_0.15_tr_0.3_pr_1.0_0/args.txt
