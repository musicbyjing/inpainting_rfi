# inpainting_rfi

A project to inpaint Radio Frequency Interference in radio telescope data. For PHYS 489 Winter 2021, McGill University.

## Requirements:

To run on Compute Canada, follow these steps to create a virtual environment with all necessary dependencies:
```
module load python/3.7.0
virtualenv ~/HERA_ENV
source ~/HERA_ENV/bin/activate
pip install git+https://github.com/HERA-Team/hera_sim
module load scipy-stack
pip install --user --no-index tensorflow_cpu
```
Then, create or import job files, and you're good to go!

When logging in again, make sure to call
```
source ~/HERA_ENV/bin/activate
module load scipy-stack
```
*BEFORE* running jobs!
