#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=0-06:30
#SBATCH --mem=128G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=%j.out

source ~/HERA_ENV/bin/activate
module load scipy-stack
python3 scripts/load_real_data.py --dim 512