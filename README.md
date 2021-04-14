# inpainting_rfi

A project to inpaint radio frequency interference (RFI) in telescope data. For PHYS 489 Winter 2021, McGill University.

## Requirements:

### Create an environment 

To run on Compute Canada, follow these steps to create a virtual environment with all necessary dependencies:
```
module load python/3.8.2
virtualenv ~/HERA_ENV
source ~/HERA_ENV/bin/activate
pip install -r requirements.txt --no-index
```
Then, create or import job files, and you're good to go!

### Load an existing environment

When logging in again, make sure to call
```
source ~/HERA_ENV/bin/activate
module load scipy-stack
```
*BEFORE* running jobs--or alternately, include these lines in the job script.

### Workflow

#### Creating a simulated dataset

1. Run `gen_data.sh`, creating a simulated dataset with simulated masks.
2. Run `crop_data.sh` to crop the dataset to squares.

#### Creating a real dataset

1. Run `load_real_data.sh` to cut a `pyuvdata` file into as many `dim`x`dim` squares as possible, stored in `{x}real_samples_{dim}x{dim}.npy`. These real visibility plots will have real RFI masks.
2. Pass the above's output to `gen_data.sh` with the `--from-vis` flag, which creates a dataset that adds on simulated masks.

#### Training

1. Run the finished dataset through `train_unet.sh`.
2. See results using `predict.sh`.
