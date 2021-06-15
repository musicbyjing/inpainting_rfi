#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=00:010:00
#SBATCH --mem=128G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=gen_vis-%j.out

source ~/HERA_ENV_2/bin/activate
# module load scipy-stack
python3 scripts/gen_data/generate_vis.py --n_examples 555 --t_rx 5000.