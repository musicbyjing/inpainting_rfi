#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=0-00:30
#SBATCH --gres=gpu:p100l:4
#SBATCH --mem=64G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
module load cuda cudnn
python3 scripts/train_model.py --max_epochs 1 --id 1612730034_10_examples_1_masks