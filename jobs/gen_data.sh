#!/bin/bash
#SBATCH --account=rrg-acliu
#SBATCH --time=00:59:00
#SBATCH --mem=128G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=gen_data-%j.out

source ~/HERA_ENV_2/bin/activate
# module load scipy-stack
# python3 scripts/gen_data/generate_dataset.py --n_sim_masks 5 --existing_vis data_real/544real_samples_512x512.npy
# python3 scripts/gen_data/generate_dataset.py --n_sim_masks 5 --existing_vis data_real/550_sim_examples_5_masks_CROPPED_512x512x3_data.npy
# python3 scripts/gen_data/generate_dataset.py --n_examples 11 --n_sim_masks 5
# python3 scripts/gen_data/generate_dataset.py --n_sim_masks 5 --existing_vis data_real/vis_list_sim_544_with_applied_real_masks.npy
python3 scripts/gen_data/generate_dataset.py --existing_vis visibilities/1623273977_555_examples_visibilities_trx5000.npy --existing_masks data/masks_from_1616898929_544_examples_5_masks.npy --three-channels