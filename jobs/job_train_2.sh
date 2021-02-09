#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=0-33:00
#SBATCH --gres=gpu:p100l:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=%N-%j.out

hostname
nvidia-smi
source ~/HERA_ENV/bin/activate
module load scipy-stack
module load cuda cudnn
python3 scripts/train_model.py --max_epochs 500 --id 1612458166_500_examples_1_masks