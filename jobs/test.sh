#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=0-03:30
#SBATCH --mem=256G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=test-%j.out

source ~/HERA_ENV_2/bin/activate
# pip install pyuvdata
module load scipy-stack
python3 scripts/test.py