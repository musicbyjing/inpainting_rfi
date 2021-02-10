#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=0-24:00
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=128G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=%N-%j.out

source ~/HERA_ENV/bin/activate
module load scipy-stack
module load cuda cudnn
python3 scripts/train_model.py --max_epochs 500 --id 1612458166_500_examples_1_masks --model colab
