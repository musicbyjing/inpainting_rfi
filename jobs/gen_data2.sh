#!/bin/bash
#SBATCH --account=rrg-acliu
#SBATCH --time=00:59:00
#SBATCH --mem=128G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=gen_data-%j.out

source ~/HERA_ENV_2/bin/activate
# module load scipy-stack
python3 scripts/gen_data/generate_dataset.py --existing_vis visibilities/1623359763_555_examples_544_masks_trx5000_dataset.npy --n_sim_masks 5