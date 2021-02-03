#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=1:00:00
#SBATCH --array=1-10
python3 scripts/generate_dataset.py --n_masks 1 --n_examples 500