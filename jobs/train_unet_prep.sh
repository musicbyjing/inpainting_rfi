#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=0-00:10
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=128G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=%N-%j.out

hostname
nvidia-smi
source ~/HERA_ENV/bin/activate
module load scipy-stack
module load cuda cudnn
python3 scripts/train_model.py --model unet --max_epochs 1 --id 1612730034_10_examples_1_masks_CROPPED_256x256 --batch_size 4