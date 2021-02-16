#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=00:59:00
#SBATCH --mem=32G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL

source ~/HERA_ENV/bin/activate
module load scipy-stack
python3 scripts/generate_dataset.py --n_masks 1 --n_examples 500
