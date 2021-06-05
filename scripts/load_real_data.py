from pyuvdata import UVData
import numpy as np
import matplotlib.pyplot as plt
import argparse
import copy, os, itertools, inspect
from tqdm import tqdm
from utils import plot_one_vis

def load_data(filename):
    uvd = UVData()
    uvd.read(filename) # By default, it would load all the baselines 
    print(uvd.data_array.shape) # (ntime*nbl, 1, nfreq, npol), so (182868, 1, 1024, 4)
    antpairpols = uvd.get_antpairpols() # all the baselines and polarizations in the file 

    # check LSTs and freqs
    # all_lsts = []
    # for key in antpairpols:
    #     lsts = uvd.get_times(key)
    #     all_lsts.append((np.max(lsts), np.min(lsts)))
    # print(all_lsts)
    # print("MEAN", np.mean(np.array(all_lsts)))

    # LST = np.unique(uvd.lst_array) * 12/np.pi #RA in hours
    # print(LST)
    # print(np.amax(LST) - np.amin(LST))
    # print(uvd.lst_array)
    # freq = uvd.freq_array[0] #freq in Hz
    # print(np.amax(freq) - np.amin(freq))
    # print(freq)


    print(f"Loaded data with {len(antpairpols)} antpairpols", flush=True)
    key = (37, 38, 'ee')
    image = uvd.get_data(key)
    mask = uvd.get_flags(key)

    plot_one_vis(image, 2.5, 3, (7,7), "title", os.path.join("images", "sample_real_phase.png"))

    # np.save("image1.npy", image)
    # np.save("mask1.npy", mask)
    return uvd, antpairpols


def crop_data(uvd, antpairpols, dim):
    ''' Crop each image to as many dim x dim images as possible '''
    data = []
    print("beginning to crop data")
    for i, key in enumerate(antpairpols):
        print(f"Starting {key}", flush=True)
        image = None
        mask = None
        image = uvd.get_data(key)
        mask = uvd.get_flags(key)
        og_height, og_width = image.shape
        
        # get bounding mask; if we can't get it, skip and go to the next key
        hor_start, hor_end = get_bounds(image, mask, og_height, og_width, isRow=True)
        if hor_start == -1 or hor_end == -1:
            print(f"Issue encountered with horizontal bounds. Skipping key {key}.", flush=True)
            continue
        vert_start, vert_end = get_bounds(image, mask, og_height, og_width)
        if vert_start == -1 or vert_end == -1:
            print(f"Issue encountered with vertical bounds. Skipping key {key}.", flush=True)
            continue
        
        # calculate n_reps and take cuts
        data = take_cuts(data, image, mask, hor_start, hor_end, vert_start, vert_end, dim)
        
        # checkpoint
        if i % 10 == 0:
            np.save(f"{len(data)}real_samples_{dim}x{dim}_iteration_{i}.npy", np.array(data))
        print(f"Finished key {key}. Iteration {i}/{len(antpairpols)}", flush=True)

    print("Exiting crop_data()...", flush=True)
    print("Total number of data points:", len(data))
    np.save(f"{len(data)}real_samples_{dim}x{dim}.npy", np.array(data))


def get_bounds(image, mask, og_height, og_width, isRow=False):
    '''
    For a single image, return coordinates of the average bounding masked rectangle.
    Do this by taking the average of the bounding boxes at several random indices.
    '''
    starts = []
    ends = []
    counter = 0
    print("Entering get_bounds()...", flush=True)
    while (len(starts) < 6) and (counter < 10): # either we get 6 values to take a mean from, or we iterate 10 times
        if isRow:
            index = np.random.randint(og_height)
            row = mask[index]     
        else:
            index = np.random.randint(og_width)
            row = mask[:, index]
        # print(index)
        spans = [(key, sum(1 for _ in group)) for key, group in itertools.groupby(row)]
        # print(spans)
        if len(spans) == 1 or len(spans) == 0:
            counter += 1
            continue
        starts.append(spans[0][1])
        ends.append(spans[-1][1])
        counter += 1
        # print("starts", len(starts), flush=True)
        # print(counter, flush=True)
    if counter == 10 and (len(starts) == 0 or len(ends) == 0):
        return -1, -1
    # print("Starts", starts, flush=True)
    # print("Ends", ends, flush=True)
    arr_start = round(np.mean(starts))
    arr_end = og_width - round(np.mean(ends)) if isRow else og_height - round(np.mean(ends))
    return arr_start, arr_end


def take_cuts(data, image, mask, hor_start, hor_end, vert_start, vert_end, dim):
    ''' Crop each image and mask to n_reps_hor by n_reps_vert examples '''
    n_reps_hor = (hor_end - hor_start) // dim
    n_reps_vert = (vert_end - vert_start) // dim

    for j in range(n_reps_vert):
        for i in range(n_reps_hor):
            example = np.zeros((dim,dim,3))
            
            one_vis = image[vert_start+j*dim : vert_start+(j+1)*dim, hor_start+i*dim : hor_start+(i+1)*dim]
            one_mask = mask[vert_start+j*dim : vert_start+(j+1)*dim, hor_start+i*dim : hor_start+(i+1)*dim]
            example[:, :, 0] = one_vis.real
            example[:, :, 1] = one_vis.imag
            example[:, :, 2] = one_mask
            
            data.append(example)
    
    return data


def crop_data_test(id, dim):
    ''' TEST FUNCTION WITH SINGLE IMAGE  '''
    image = np.load(f"data_real/image{id}.npy")
    mask = np.load(f"data_real/mask{id}.npy")
    data = []
    og_height, og_width = image.shape
    print("OG shape", image.shape)
    
    # get bounding mask
    hor_start, hor_end = get_bounds(image, mask, og_height, og_width, isRow=True)
    vert_start, vert_end = get_bounds(image, mask, og_height, og_width)
    if hor_start == -1 or hor_end == -1 or vert_start == -1 or vert_end == -1: # skip and go to the next key
        print("Issue encountered! Skipping this key.")

    # calculate n_reps and take cuts
    data = take_cuts(data, image, mask, hor_start, hor_end, vert_start, vert_end, dim)
    
    data = np.array(data)
    print("Data shape:", data.shape)
    np.save(f"{data.shape[0]}real_samples_{dim}x{dim}.npy", data)


def main():
    # cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, help="Size of one dimension (return square data)")
    parser.add_argument("--no-save", default=True, action="store_false", help="Use this flag to run tests without saving generated files to disk")
    args = parser.parse_args()

    dim = args.dim
    save = args.no_save
    
    # REAL
    print("starting", flush=True)
    filename = os.path.join('data_real', 'sample.uvh5')
    uvd, antpairpols = load_data(filename)

    # crop_data(uvd, antpairpols, dim)
    
    # TEST
    # crop_data_test('3738ee', dim)

    print("load_real_data.py complete.", flush=True)

    
if __name__ == "__main__":
    main()