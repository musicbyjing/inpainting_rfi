#!/bin/bash
#SBATCH --account=def-acliu
#SBATCH --time=0-00:10
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --mail-user=jing.liu6@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --output=%N-%j.out

source ~/HERA_ENV/bin/activate
module load scipy-stack
module load cuda cudnn
python3 scripts/train_model.py --max_epochs 0 --id 0 --compile_only