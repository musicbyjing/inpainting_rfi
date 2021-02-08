# inpainting_rfi

A project to inpaint radio frequency interference (RFI) in telescope data. For PHYS 489 Winter 2021, McGill University.

## Requirements:

To run on Compute Canada, follow these steps to create a virtual environment with all necessary dependencies:
```
module load python/3.7.0
virtualenv ~/HERA_ENV
source ~/HERA_ENV/bin/activate
pip install git+https://github.com/HERA-Team/hera_sim
pip install git+https://github.com/HERA-Team/uvtools
module load scipy-stack
pip install --user --no-index tensorflow_gpu
```
Then, create or import job files, and you're good to go!

When logging in again, make sure to call
```
source ~/HERA_ENV/bin/activate
module load scipy-stack
```
*BEFORE* running jobs--or alternately, include these lines in the job script.
