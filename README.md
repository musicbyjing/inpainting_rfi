# inpainting_rfi

A project to inpaint radio frequency interference (RFI) in telescope data. For PHYS 489 Winter 2021, McGill University.

# Requirements:

## Create an environment 

To run on Compute Canada, follow these steps to create a virtual environment with all necessary dependencies:
```
module load python/3.8.2
virtualenv --no-download ~/HERA_ENV
source ~/HERA_ENV/bin/activate
pip install -r requirements.txt
pip install --no-index tensorflow
```
Then, create or import job files, and you're good to go!

## Load an existing environment

When logging in again, make sure to call
```
source ~/HERA_ENV/bin/activate
```
*BEFORE* running jobs--or alternately, include these lines in the job script.

(To deactivate a virtual environment, simply call `deactivate`.)

# Terminology

- An **RFI mask** (I use "mask" and "flag" interchangeably throughout this code) refers to the binary 2D array that denotes where RFI exists. Within a mask, 1 &rarr; RFI exists, 0 &rarr; no RFI. (Accordingly, `data[mask == True] = 0` sets `data` to 0 where the RFI exists.)
    - Masks can be *simulated* (generated) or *real* (from HERA)
- A training **dataset** is composed of two files: `/data/ID_dataset.npy` and `/data/ID_labels.npy`, where `ID` is the same. When running scripts on a particular dataset, the usage is `<command> --id ID`.

# Workflow

## Creating a simulated dataset

1. Run `gen_data.sh` to create a simulated dataset with simulated masks.
2. Run `crop_data.sh` to crop the dataset to squares for easier training.

## Creating a real dataset

1. Run `load_real_data.sh` to cut a `pyuvdata` file into as many `dim` x `dim` squares as possible, stored in `{n}real_samples_{dim}x{dim}.npy`. These real visibility plots will have real RFI masks.
2. Pass the above's output to `gen_data.sh` with the `--from-vis` flag, which creates a dataset that adds on simulated masks.

## Training

1. Run the finished dataset through `train_unet.sh`.
2. See results using `predict.sh`.

# Overview of files

## `/scripts/`

- `crop_existing_dataset.py`: Given a dataset {data, labels} where each example is F (freq) x T (time), crops into D x D squares by taking the first D pixels in each dimension.
    - Usage: `python3 scripts/crop_existing_dataset.py --dim D --id ID`
- `cut_existing_dataset.py`: Given a dataset where each example is F x T, divides each example into AB examples where each new example is F//B x T//A, where // denotes integer division.
    - Usage: `python3 scripts/cut_existing_dataset.py --id ID --divide-time-by A --divide-freq-by B`
- `load_real_data.py`: Loads real data and cut each image into as many D x D squares as possible.
    - Usage: `python3 scripts/load_real_data.py --dim D`
- `predict.py`: Takes a random image from the dataset specified by ID and generates predictions using MODEL. Use `--no-ground-truth` flag if running on real data.
    - Usage: `python3 scripts/predict.py --id ID --model-name MODEL [--no-ground-truth]`
- `train_model.py`: Trains an ML model on data.
    - Usage: `python3 scripts/train_model.py --model [unet/colab/alex] --max_epochs E --id ID --batch_size B [--normalize]`

- `models_X.py`: ML models
- `utils.py`: utilities

### `gen_data/`

- `generate_dataset.py`: Generates a dataset for training. There are two components to a dataset: visibilities and masks.
    1. Load a real HERA mask and get its dimensions (`get_dims_real_mask.py`).
    2. Obtain masks
        
        - If we have an existing mask list, use the `--existing_masks` flag to specify its location.
        
        - Otherwise, generate simulated masks using the above dimensions (`generate_masks.py`).
    3. Obtain visibilities

        - If we have existing visibility plot list, use the `--existing_vis` flag to specify its location.

        - Otherwise, generate simulated visibilities using the above dimensions (`generate_vis.py`).
    
    4. Apply the masks to the visibilities, resulting in a dataset that has dimensions (n_examples, times, freqs, 4). The last dimension has the following channels: (real, imag, M, P) where M is a binary mask representing areas with missing data in channels 1 and 2, and P is the prediction area. 
        - In cases where a simulated mask is applied over top of real data (with real masks), M is the area where the real data is 0, while the P is the simulated mask - real mask.
    
    - Usage: 
        - `python3 scripts/gen_data/generate_dataset.py --n_sim_masks M --n_examples N` generates N *simulated* visibilities with M *simulated* masks applied at random.
        - `python3 scripts/gen_data/generate_dataset.py --n_sim_masks M --existing_vis data_real/DATA.npy` where `DATA.npy` holds existing visibilities. This usage applies M simulated masks at random over top of existing visibilities.
        - `python3 scripts/gen_data/generate_dataset.py --existing_vis data_real/DATA.npy --existing_masks data_real/MASK.npy` applies masks from `MASK.npy` at random to visibilities in `DATA.npy`.

# Troubleshooting

- (May 10, 2021) ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject
    - `pip install numpy==1.20`