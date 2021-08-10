from pyuvdata import UVData
import numpy as np
import matplotlib.pyplot as plt
import argparse
import copy, os, itertools, inspect
from tqdm import tqdm
from utils import plot_one_vis
import random
from itertools import permutations
import time
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

def save_sets(key, night1, night2, night3, night4, night5):
    '''
    Make plots from 1 key for 5 different nights of data
    '''
    print(key)

    np.save(f"images/{key}_night1.npy", night1.get_data(key))
    np.save(f"images/{key}_night2.npy", night2.get_data(key))
    np.save(f"images/{key}_night3.npy", night3.get_data(key))
    np.save(f"images/{key}_night4.npy", night4.get_data(key))
    np.save(f"images/{key}_night5.npy", night5.get_data(key))

    print(f"{key} complete.")



def apply_shifts(images, shifts):
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
        res.append(img_new)
    return res
    


def main():
    # cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=512, help="Size of one dimension (return square data)")
    parser.add_argument("--no-save", default=True, action="store_false", help="Use this flag to run tests without saving generated files to disk")
    parser.add_argument("--num-antpairpols", type=int, default=168)
    args = parser.parse_args()

    dim = args.dim
    save = args.no_save
    num_antpairpols = args.num_antpairpols

    print("starting", flush=True)

    time1 = time.time()
    all_sets = np.load("sets/ALL_SETS.npy")
    print("ALL SETS:", all_sets.shape)
    shifts = [0, 110, 214, 273, 339] # the means of FFT-calculated shifts for all keys, after outliers > 3 sigma's removed
    
    all_shifted_sets = []

    for set_ in all_sets[:2]:
        
        shifted_set = apply_shifts(set_, shifts)
        all_shifted_sets.append(shifted_set)

        # # Find max positive shift and min negative shift
        # # and crop all images in the set
        # max_shift = max(shifts)
        # min_shift = min(shifts)

        # time9 = time.time()
        # if min_shift < 0 and max_shift > 0:
        #     for img in image_set:
        #         img = img[-min_shift:-max_shift]
        # elif max_shift > 0 and min_shift >= 0:
        #     for img in image_set:
        #         img = img[:-max_shift]
        # elif max_shift <= 0 and min_shift < 0:
        #     for img in image_set:
        #         img = img[-min_shift:]
        # time10 = time.time()
        # print(f"TIME ELAPSED: {time10-time9} s", flush=True)

        # all_sets.append(shifted_images)
        # np.save(f"sets/{key}_shifted.npy", shifted_images)
        
        np.save(f"sets/{key}_shifted.npy", shifted_set)

    np.save(f"sets/ALL_SHIFTED.npy", np.array(all_shifted_sets))
    print("shift_sets.py complete.", flush=True)

    
if __name__ == "__main__":
    main()