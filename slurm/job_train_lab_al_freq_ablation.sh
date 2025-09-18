#!/bin/sh
 
#SBATCH --job-name=train_lab_al_linear
#SBATCH --output=slurm/logs/train_lab_al_linear.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=slurm/logs/train_lab_al_linear.err  # Standard error of the script
#SBATCH --partition=students
#SBATCH --time=2-00:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=12  # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=48G  # Memory in GB (Don't use more than 48/24 per GPU unless you absolutely need it and know what you are doing)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ultranba

python run_ultranba.py --expname "lab_al_barf_no_freq_ab" --config configs/config_lab_al_freq.txt --i_weights 10000 --reg --tensorboard --no_freq_adjustment
python run_ultranerf.py --expname "lab_al_nerf_ab" --config configs/config_lab_al_freq.txt --i_weights 10000 --reg --tensorboard --multires 16
python run_ultranba.py --expname "lab_al_barf_ab" --config configs/config_lab_al_freq.txt --i_weights 10000 --reg --tensorboard
