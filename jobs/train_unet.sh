#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=0-20:00
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=128G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=unet-%j.out

hostname
nvidia-smi
source ~/HERA_ENV_3.8.2/bin/activate
# module load scipy-stack
module load cuda cudnn
python3 scripts/train_model.py --model unet --max_epochs 5000 --id 1617565338_550_examples_5_masks --batch_size 8