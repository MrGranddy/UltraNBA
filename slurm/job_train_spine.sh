#!/bin/sh
 
#SBATCH --job-name=train_spine
#SBATCH --output=slurm/logs/train_spine.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=slurm/logs/train_spine.err  # Standard error of the script
#SBATCH --account=students
#SBATCH --partition=students
#SBATCH --time=1-00:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=12  # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=48G  # Memory in GB (Don't use more than 48/24 per GPU unless you absolutely need it and know what you are doing)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ultranba
python run_ultranerf.py --expname "nerf_reg_spine_1" --config configs/config_base_nerf_spine.txt --i_weights 20000 --reg --tensorboard --lrate 1.e-4
python run_ultranerf.py --expname "nerf_reg_spine_2" --config configs/config_base_nerf_spine.txt --i_weights 20000 --reg --tensorboard --lrate 1.e-5
python run_ultranerf.py --expname "nerf_spine_1_no_reg" --config configs/config_base_nerf_spine.txt --i_weights 20000 --tensorboard --lrate 1.e-4
python run_ultranerf.py --expname "nerf_spine_2_no_reg" --config configs/config_base_nerf_spine.txt --i_weights 20000 --tensorboard --lrate 1.e-5
