#!/bin/bash
#SBATCH --account=rrg-acliu
#SBATCH --time=00:59:00
#SBATCH --mem=128G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=gen_data-%j.out

source ~/HERA_ENV/bin/activate
module load scipy-stack
# python3 scripts/generate_dataset.py --n_masks 5 --from-vis data_real/544real_samples_512x512.npy
python3 scripts/generate_dataset.py --n_masks 5 --from-vis data_real/550_sim_examples_5_masks_CROPPED_512x512x3_data.npy
# python3 scripts/generate_dataset.py --n_masks 5 --n_examples 550