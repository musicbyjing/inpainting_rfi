#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
python3 scripts/train_model.py --max_epochs 0 --id 0 --compile_only