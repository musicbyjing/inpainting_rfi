#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=00:59:00
#SBATCH --mem=0
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=crop_data-%j.out

source ~/HERA_ENV/bin/activate
module load scipy-stack
python3 scripts/crop_existing_dataset.py --dim 512 --id 1617411003_550_examples_5_masks