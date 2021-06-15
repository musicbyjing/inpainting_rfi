#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=0-00:30
#SBATCH --mem=32G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=predict-%j.out

source ~/HERA_ENV_2/bin/activate
# module load scipy-stack
module load cuda cudnn
# python3 scripts/predict.py --id XXX --no-ground-truth --model-name unet_1616898929_544_examples_5_masks_weights_best.hdf5
# python3 scripts/predict.py --id 1616898929_544_examples_5_masks --model-name unet_1616898929_544_examples_5_masks_weights_best.hdf5
# python3 scripts/predict.py --id 1612458166_500_examples_1_masks_CROPPED_512x512 --model-name unet_1612458166_500_examples_1_masks_CROPPED_512x512_model.h5
python3 scripts/predict.py --id 1623596629_555_examples_5_masks_trx1500 --model-name unet_1623596629_555_examples_5_masks_trx1500_weights_best.hdf5