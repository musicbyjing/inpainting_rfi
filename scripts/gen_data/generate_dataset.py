import numpy as np
import random
import argparse
import os.path
import h5py
import itertools
import time
import sys

from hera_sim import foregrounds, noise, sigchain, rfi, simulate
from get_dims_real_mask import *
from generate_vis import *
from generate_masks import *


##############################
##### Generate datasets #####
##############################

def create_dataset(vis_list, mask_list, save, existing_vis, three_channels):
    if vis_list.shape[-1] < 2:
        raise Exception("Need to separate visibilities into real and complex channels!")

    if three_channels:
        data = np.zeros((vis_list.shape[0], vis_list.shape[1], vis_list.shape[2], 3))
        labels = np.zeros((vis_list.shape[0], vis_list.shape[1], vis_list.shape[2], 3))
        data = vis_list # copy vis over on all three channels
        labels = vis_list # copy vis over on all three channels
    else:
        data = np.zeros((vis_list.shape[0], vis_list.shape[1], vis_list.shape[2], 4))
        labels = np.zeros((vis_list.shape[0], vis_list.shape[1], vis_list.shape[2], 4))
        data[:, :, :, :3] = vis_list # copy vis over on the first three channels
        labels[:, :, :, :3] = vis_list # copy vis over on the first three channels

    # apply random mask to each visibility plot (works for 1 mask too)
    for i, v in enumerate(data):
        sim_mask = random.choice(mask_list)

        # crop mask to fit dimensions of data, if necessary
        if sim_mask.shape != data[i].shape[:2]:
            x, y = data[i].shape[:2]
            xs = random.randint(0, sim_mask.shape[0] - x) # x start
            ys = random.randint(0, sim_mask.shape[1] - y) # y start
            sim_mask = sim_mask[xs:xs+x, ys:ys+y]
            # sim_mask = sim_mask[-x:x, ys:ys+y]
        
        print("COND0", (sim_mask == (data[i][:, :, 2])).all())
        
        data[i][:, :, :2][sim_mask == True] = 0 # Set real and imag to 0 where simulated mask exists

        if existing_vis is None: # simulated visibility plots
            data[i][:, :, 2] = sim_mask
            labels[i][:, :, 2] = sim_mask
            
            if not three_channels: # i.e. we're running with this simulated data directly
                data[i][:, :, 3] = sim_mask
                labels[i][:, :, 3] = sim_mask

        else: # real visibility plots (with real masks)
            real_mask = data[i][:, :, 2]
            mask_diff =  np.logical_and(sim_mask == True, real_mask == False).astype('uint8')

            # print("COND1", np.any(sim_mask))
            # print("COND2", np.any(mask_diff))
            # print("COND3", (sim_mask == real_mask).all())

            # print(mask_diff.shape)
            data[i][:, :, 3] = mask_diff # Set 4th channel of data as simulated mask - real mask
            labels[i][:, :, 3] = mask_diff # Set 4th channel of data as simulated mask - real mask

        # print(np.count_nonzero(train_dataset[0]==0)) # check number of 0's in a given vis (to check if mask worked)
    
    # Save files
    if save:
        prefix = f"{int(time.time())}_{len(vis_list)}_examples_{len(mask_list)}_masks"
        if existing_vis is None:
            prefix += "_sim"
        np.save(os.path.join("data", f"{prefix}_dataset_1sample.npy"), data[0])
        np.save(os.path.join("data", f"{prefix}_dataset.npy"), data)
        np.save(os.path.join("data", f"{prefix}_labels.npy"), labels)
        
        print(f"Dataset saved as {prefix}_dataset.npy")
    print(f"Data shape: {data.shape}. Labels shape: {labels.shape}.")


##############################
##### main function ##########
##############################

def main():
    # cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_examples", nargs='?', type=int, help="Number of examples to generate")
    parser.add_argument("--n_sim_masks", nargs='?', type=int,  help="Number of generated (simulated) masks")
    parser.add_argument("--no_save", default=True, action="store_false", help="Use this flag to run tests without saving generated files to disk")
    parser.add_argument("--existing_vis", nargs='?', default=None, help="Use this flag to create a dataset using existing visibilities")
    parser.add_argument("--existing_masks", nargs='?', default=None, help="Use this flag to create a dataset using existing masks")
    parser.add_argument("--t_rx", nargs='?', default=150., help="t_rx parameter in noise generation")
    # parser.add_argument("--three-channels", default=False, action="store_true", help="Output data with 3 channels only (default 4). Useful when generating simulated data that will be double masked.")
    args = parser.parse_args()

    n_masks = args.n_sim_masks
    n_examples = args.n_examples
    save = args.no_save
    existing_vis = args.existing_vis
    existing_masks = args.existing_masks
    t_rx = args.t_rx
    # three_channels = args.three_channels

    # get dimensions
    num_jd, num_reduced_channels, start_freq, end_freq, rfi_widths, rfi_heights = generate_masks_prep()
    # get masks
    if existing_masks is not None: # existing (real) masks
        mask_list = np.load(existing_masks)
    else: # need to generate masks
        mask_list = generate_masks(n_masks, num_jd, num_reduced_channels, rfi_widths, rfi_heights)

    # get visibilities
    if existing_vis is not None: # existing (real) visibilities
        vis_list = np.load(existing_vis)
    else: # need to simulate visibilities
        vis_list = generate_simulated_vis_wrapper(n_examples, num_jd, start_freq, end_freq, num_reduced_channels, False, t_rx)

    # create dataset
    create_dataset(vis_list, mask_list, save, existing_vis, False)

    print("generate_dataset.py complete.")


if __name__ == "__main__":
    main()