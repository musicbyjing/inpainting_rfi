#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=0-00:30
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=predict-%j.out

source ~/HERA_ENV_2/bin/activate
# module load scipy-stack
module load cuda cudnn
python3 scripts/predict.py --id 1623597641_555_examples_5_masks_trx5000 --model-name unet_1623597641_555_examples_5_masks_trx5000_weights_best.hdf5