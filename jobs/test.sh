#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=0-00:10
#SBATCH --mem=64G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=test-%j.out

source ~/HERA_ENV/bin/activate
module load scipy-stack
python3 scripts/test.py