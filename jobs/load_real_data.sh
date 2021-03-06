#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=0-06:30
#SBATCH --mem=128G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=load_real_data-%j.out

source ~/HERA_ENV_2/bin/activate
# module load scipy-stack
python3 scripts/load_real_data.py --dim 512