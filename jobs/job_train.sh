#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=0-05:00
#SBATCH --gres=gpu:p100l:4
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=%N-%j.out

module load cuda cudnn
python3 scripts/train_model.py --max_epochs 500 --id 1612458166_500_examples_1_masks