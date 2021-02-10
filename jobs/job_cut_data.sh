#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=00:59:00
#SBATCH --mem=32G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
python3 scripts/cut_existing_dataset.py --id 1612730034_10_examples_1_masks --divide-time-by 2 --divide-freq-by 2 --no-save
