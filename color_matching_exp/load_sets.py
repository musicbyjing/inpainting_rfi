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

def load_all_data():
    night1, antpairpols = load_data("1-2458098.uvh5", antpairpols=True)
    night2 = load_data("2-2458103.uvh5")
    night3 = load_data("3-2458108.uvh5")
    night4 = load_data("4-2458112.uvh5")
    night5 = load_data("5-2458115.uvh5")
    # the filenames (along with night 1, night 2, etc.) are arbitrary
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


def calculate_min_shift(image_set):
    ''' Slide image vertically (time axis) and return the shift that 
    minimizes the difference between the shifted image and ref
    '''
    
    ref_img = image_set[0]
    min_shifts = [0]
    for i in range(1, len(image_set)):
        
        shifts, diff = [], []
        for s in range(1, ref_img.shape[0], 2): # 4354
            # shift down by s
            shifts.append(s)
            shifted = np.pad(image_set[i], [(s,0), (0,0)], mode='constant')[:-s, :]
            diff.append(np.sum(np.abs(np.angle(shifted)-np.angle(ref_img))))
            # shift up by s
            shifts.append(-s)
            shifted = np.pad(image_set[i], [(0,s), (0,0)], mode='constant')[s:, :]
            diff.append(np.sum(np.abs(np.angle(shifted)-np.angle(ref_img))))
        
        min_shifts.append(shifts[np.argmin(diff)])
    
    return min_shifts

def calculate_shifts_fft(image_set):
    ''' Slide image vertically (time axis) and return the shift that 
    minimizes the difference between the shifted image and ref
    '''
    ref_img = image_set[0]
    f0 = fft_conv(ref_img, ref_img)
    x0 = np.argmax(np.abs(f0), axis=0)
    
    shifts = [0]
    for i in range(1, len(image_set)):
        f1 = fft_conv(image_set[i], ref_img)
        x1 = np.argmax(np.abs(f1), axis=0)
        shift = mode(x0-x1)[0][0]
        shifts.append(shift)
    
    return shifts


def fft_conv(img_to_shift, ref_img):
    ''' Slide image vertically (time axis) and return the shift that 
    minimizes the difference between the shifted image and ref
    '''
    # padding: https://stackoverflow.com/questions/28468307/scipy-ndimage-filters-convolve-and-multiplying-fourier-transforms-give-different
    assert ref_img.shape == img_to_shift.shape
    nrows, ncols = ref_img.shape
    
    ref_img = np.pad(ref_img,[(nrows-1, nrows-1), (0,0)], mode='constant')
    img_to_shift = np.pad(img_to_shift, [(nrows-1, nrows-1), (0,0)], mode='constant')

    f1 = np.fft.fft(ref_img, axis=0)
    f2 = np.fft.fft((img_to_shift), axis=0)
    
    fft_prod = f1 * f2
    idx = nrows // 3
    return np.fft.ifft(fft_prod, axis=0)[idx:idx+nrows, :]
    
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
    all_shifts = []

    for key in antpairpols[:num_antpairpols]:
        
        print(key)
        
        unshifted_images = np.array([night1.get_data(key), night2.get_data(key), night3.get_data(key), \
            night4.get_data(key), night5.get_data(key)])
        
        shifts = np.rint(calculate_shifts_fft(unshifted_images)).astype(int)
        print("shifts, fft:", shifts)
        all_shifts.append(shifts)

        # shifts_conv = np.rint(calculate_min_shift(unshifted_images)).astype(int)
        # print("shifts, conv:", shifts_conv)
        
        all_sets.append(unshifted_images)
        
        np.save(f"sets/{key}_unshifted.npy", unshifted_images)

    np.savetxt("sets/all_shifts.csv", all_shifts, delimiter=",")

    all_sets = np.array(all_sets)
    np.save(f"sets/ALL_SETS.npy", all_sets)
    print("ALL_SETS.npy saved with shape ", all_sets.shape)
    print("load_real_data.py complete.", flush=True)

    
if __name__ == "__main__":
    main()