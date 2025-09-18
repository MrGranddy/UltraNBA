#!/bin/sh
 
#SBATCH --job-name=generate_perturbed_poses
#SBATCH --output=slurm/logs/generate_perturbed_poses.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=slurm/logs/generate_perturbed_poses.err  # Standard error of the script
#SBATCH --account=students
#SBATCH --partition=students
#SBATCH --time=0-12:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:0  # Number of GPUs if needed
#SBATCH --cpus-per-task=2  # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=32G  # Memory in GB (Don't use more than 48/24 per GPU unless you absolutely need it and know what you are doing)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ultranba
# python generate_perturbed_noise.py --data_dir /home/guests/{NAME}/ultrasound_data/original/spine
# python generate_perturbed_noise.py --data_dir /home/guests/{NAME}/ultrasound_data/original/liver
# python generate_perturbed_noise.py --data_dir /home/guests/{NAME}/ultrasound_data/original/LAB_AL
python generate_perturbed_noise.py --data_dir /home/guests/{NAME}/ultrasound_data/original/LAB_ALx2
python generate_perturbed_noise.py --data_dir /home/guests/{NAME}/ultrasound_data/original/WAN_IK
python generate_perturbed_noise.py --data_dir /home/guests/{NAME}/ultrasound_data/original/WAN_IKx2