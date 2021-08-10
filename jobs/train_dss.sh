#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=0-01:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=train_dss-%j.out

source ~/HERA_ENV_2/bin/activate
# module load scipy-stack
module load cuda cudnn
python3 color_matching_exp/run.py