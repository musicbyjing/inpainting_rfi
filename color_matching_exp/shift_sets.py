from pyuvdata import UVData
import numpy as np
import matplotlib.pyplot as plt
import argparse
import copy, os, itertools, inspect
from tqdm import tqdm
from utils import plot_one_vis
import random
from itertools import permutations
from scipy.stats import mode


def load_data(filename, antpairpols=False):
    '''
    Load real dataset from uvh5 format
    '''
    folder = "data_real"
    
    uvd = UVData()
    uvd.read(os.path.join(folder, filename)) # By default, it would load all the baselines 
    print(f"Loaded data with shape {uvd.data_array.shape}", flush=True)  # (ntime*nbl, 1, nfreq, npol), so (182868, 1, 1024, 4)
    if antpairpols:
        antpairpols = uvd.get_antpairpols() # all the baselines and polarizations in the file 
        return uvd, antpairpols
    return uvd


def apply_shifts(images, shifts, max_shift, min_shift):
    '''
    Apply shifts to images and return a new array of shifted images
    '''
    res = []

    for image, shift in zip(images, shifts):
        if shift == 0:
            img_new = image
        elif shift < 0:
            shift = -shift
            img_new = np.pad(image, [(0,shift), (0,0)], mode='constant')[shift:, :]
        else:
            img_new = np.pad(image, [(shift,0), (0, 0)], mode='constant')[:-shift, :]
        
        # Crop each image
        # This is the maximally-sized usable shifted set. May have to crop into a smaller square to use with the DSS network
        img_new = img_new[max_shift:, :]

        res.append(img_new)
    return res
    


def main():
    print("starting", flush=True)

    all_sets = np.load("sets/ALL_SETS.npy")
    print("ALL SETS:", all_sets.shape)
    shifts = [0, 110, 214, 273, 339] # the means of FFT-calculated shifts for all keys, after outliers > 3 sigma's removed
    
    all_shifted_sets = []

    for i, set_ in enumerate(all_sets):
        
        shifted_set = apply_shifts(set_, shifts, max(shifts), min(shifts))
        all_shifted_sets.append(shifted_set)
        
        np.save(f"sets/set{i}_shifted.npy", np.array(shifted_set))

    np.save(f"sets/ALL_SHIFTED.npy", np.array(all_shifted_sets))
    print("shift_sets.py complete.", flush=True)

    
if __name__ == "__main__":
    main()