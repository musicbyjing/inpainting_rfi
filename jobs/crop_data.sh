#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=00:59:00
#SBATCH --mem=128G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
source ~/HERA_ENV/bin/activate
module load scipy-stack
python3 scripts/crop_existing_dataset.py --dim 512 --id 1612458166_500_examples_1_masks