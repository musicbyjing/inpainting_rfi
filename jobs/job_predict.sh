#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=0-00:30
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=%N-%j.out

source ~/HERA_ENV/bin/activate
module load scipy-stack
module load cuda cudnn
python3 scripts/predict.py --id 1612458166_500_examples_1_masks --model-name models/123.87991_colab_weights.best.hdf5