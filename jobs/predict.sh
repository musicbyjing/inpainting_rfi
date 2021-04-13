#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=0-00:30
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=predict-%j.out

source ~/HERA_ENV/bin/activate
module load scipy-stack
module load cuda cudnn
# python3 scripts/predict.py --id 6real_samples_512x512_4d --no-ground-truth --model-name unet_1616255586_544_examples_5_masks_weights_best.hdf5
python3 scripts/predict.py --id 1616984161_550_sim_examples_5_masks_CROPPED_512x512 --model-name unet_1616984161_550_sim_examples_5_masks_CROPPED_512x512_weights_best.hdf5