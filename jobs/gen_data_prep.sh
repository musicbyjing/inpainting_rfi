#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=00:010:00
#SBATCH --mem=4G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL

source ~/HERA_ENV/bin/activate
module load scipy-stack
python3 scripts/generate_dataset.py --n_masks 3 --n_examples 5
