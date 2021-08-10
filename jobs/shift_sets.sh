#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=00-60:00
#SBATCH --mem=256G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=shift_sets-%j.out

source ~/HERA_ENV_2/bin/activate
# module load scipy-stack
python3 scripts/shift_sets.py