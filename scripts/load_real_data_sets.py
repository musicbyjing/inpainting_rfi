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

def load_all_data():
    night1, antpairpols = load_data("1-2458098.uvh5", antpairpols=True)
    night2 = load_data("2-2458103.uvh5")
    night3 = load_data("3-2458108.uvh5")
    night4 = load_data("4-2458112.uvh5")
    night5 = load_data("5-2458115.uvh5")
    return night1, night2, night3, night4, night5, antpairpols

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
    Make plots from 1 key for 2 different nights of data
    '''
    print(key)

    np.save(f"images/{key}_night1.npy", night1.get_data(key))
    np.save(f"images/{key}_night2.npy", night2.get_data(key))
    np.save(f"images/{key}_night3.npy", night3.get_data(key))
    np.save(f"images/{key}_night4.npy", night4.get_data(key))
    np.save(f"images/{key}_night5.npy", night5.get_data(key))

    print(f"{key} complete.")


def calculate_min_shift(img_to_shift, ref_img):
    ''' Slide image vertically (time axis) and return the shift that 
    minimizes the difference between the shifted image and ref
    '''
    shifts, diff = [], []
    for s in range(1, ref_img.shape[0], 2): # 4354
        # shift down by s
        shifts.append(s)
        shifted = np.pad(img_to_shift, [(s,0), (0,0)], mode='constant')[:-s, :]
        diff.append(np.sum(np.abs(shifted-ref_img)))
        # shift up by s
        shifts.append(-s)
        shifted = np.pad(img_to_shift, [(0,s), (0,0)], mode='constant')[s:, :]
        diff.append(np.sum(np.abs(shifted-ref_img)))
        
    return shifts[np.argmin(diff)]

def calculate_min_shift_fft(img_to_shift, ref_img):
    ''' Slide image vertically (time axis) and return the shift that 
    minimizes the difference between the shifted image and ref
    '''
    return np.amax(np.fft.ifftn(np.fft.fftn(img_to_shift) * np.conj(np.fft.fftn(ref_img))))



def apply_min_shift(img_to_shift, shift):
    '''
    Apply min shift to img_to_shift and return
    '''
    if shift < 0:
        shift = -shift
        img_new = np.pad(img_to_shift, [(0,shift), (0,0)], mode='constant')[shift:, :]
    else:
        img_new = np.pad(img_to_shift, [(shift,0), (0, 0)], mode='constant')[:-shift, :]
    return img_new


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
    night1, night2, night3, night4, night5, antpairpols = load_all_data()
    
    time2 = time.time()
    print(f"LOAD ALL DATA: {time2-time1} s", flush=True)
    nights = np.array([night2, night3, night4, night5])
    
    time3 = time.time()
    print(f"CREATE NIGHTS ARRAY: {time3-time2} s", flush=True)

    all_sets = []

    for key in antpairpols[:num_antpairpols]:
        
        print(key)
        
        ref_img = night1.get_data(key) # image 1
        unshifted_set = [ref_img]
        image_set = [ref_img]
        shifts = [0]

        for night in nights:
            time4 = time.time()
            img = night.get_data(key)
            unshifted_set.append(img)
            '''
            time5 = time.time()
            print(f"GET ONE KEY'S DATA: {time5-time4} s", flush=True)
            shift = calculate_min_shift(img, ref_img)
            time6 = time.time()
            print(f"CALCULATE SHIFT FOR ONE KEY: {time6-time5} s", flush=True)
            img = apply_min_shift(img, shift)
            time7 = time.time()
            print(f"APPLY SHIFT FOR ONE KEY: {time7-time6} s", flush=True)
            image_set.append(img)
            shifts.append(shift)
            time8 = time.time()
            print(f"APPEND SHIFT, IMAGE TO COLLECTION: {time8-time7} s", flush=True)
        
        print("shifts:", shifts)

        # Find max positive shift and min negative shift
        # and crop all images in the set
        max_shift = max(shifts)
        min_shift = min(shifts)

        time9 = time.time()
        if min_shift < 0 and max_shift > 0:
            for img in image_set:
                img = img[-min_shift:-max_shift]
        elif max_shift > 0 and min_shift >= 0:
            for img in image_set:
                img = img[:-max_shift]
        elif max_shift <= 0 and min_shift < 0:
            for img in image_set:
                img = img[-min_shift:]
        time10 = time.time()
        print(f"TIME ELAPSED: {time10-time9} s", flush=True)

        all_sets.append(image_set)
        np.save(f"sets/{key}_sets.npy", image_set)
        '''
        np.save(f"sets/{key}_unshifted.npy", unshifted_set)

    
    # np.save(f"sets/ALL_SETS.npy", all_sets)
    print("load_real_data.py complete.", flush=True)

    
if __name__ == "__main__":
    main()