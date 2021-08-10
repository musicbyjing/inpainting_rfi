#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=0-20:00
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=128G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=train_unet_2-%j.out

hostname
nvidia-smi
source ~/HERA_ENV_2/bin/activate
# module load scipy-stack
module load cuda cudnn
python3 scripts/train_model.py --model unet --max_epochs 1000 --id 1624637389_216_examples_5_masks_MOD --batch_size 8 --normalize all