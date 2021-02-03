#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pylab as plt
import random
import argparse
import os.path
import h5py
import itertools
import time

from hera_sim import foregrounds, noise, sigchain, rfi, simulate

##############################
### Generate custom masks ###
##############################

def load_real_mask(filename):
    '''
    Load a real mask (e.g. from HERA)
    '''
    f = h5py.File(filename, "r")
    saurabhs_mask = f['mask'] # the "actual" mask
    num_jd = saurabhs_mask.shape[0]
    num_freq_channels = saurabhs_mask.shape[1]

    return saurabhs_mask, num_jd, num_freq_channels

def get_RFI_spans(row, isRow=False):
    '''
    Take horizontal or vertical pixel slice of the mask. Spans of the `True` sections = spans of RFI
        if isRow, remove the large RFI caps at either end
    '''
    spans = [(key, sum(1 for _ in group)) for key, group in itertools.groupby(row)]
    if len(spans) == 1:
        raise Exception("Error: all values in the row/col are True; select another one")
    if isRow:
        start = spans[0][1]
        end = spans[-1][1]
        print("RFI bookends", start, end)
        spans = spans[1:-1]
        return spans, start, end
    else:
        return spans

def generate_one_mask(time, freq, widths, heights):
    '''
    Generate random RFI mask with dimensions time x freq, using RFI spans from a real mask
    '''
    random.shuffle(widths)
    random.shuffle(heights)
    
    # row by row
    one_row = []
    for w in widths:
        one_row.extend([w[0]] * w[1])
    mask = np.tile(one_row, (time, 1)) # copy one_row `time` times in the vertical direction
    
    # col by col
    one_col = []
    for w in heights:
        one_col.extend([w[0]] * w[1])
    mask2 = np.tile(np.array(one_col).reshape((time, 1)), (1, freq))
    
    combined_mask = np.logical_or(mask, mask2) # any cell with True will have RFI
    return combined_mask

def generate_masks(n, time, freq, widths, heights):
    '''
    Generate n masks
    '''
    return np.array([generate_one_mask(time, freq, widths, heights) for i in range(n)])


##############################
##### Generate vis plots #####
##############################

def generate_one_vis_plot(lsts, fqs, bl_len_ns):
    '''
    Generate one visibility waterfall plot
    Returns a 2D (x,y) plot, as well as a channel-separated plot (x,y,2)
    '''
    # point-source and diffuse foregrounds
    Tsky_mdl = noise.HERA_Tsky_mdl['xx']
    vis = foregrounds.diffuse_foreground(lsts, fqs, bl_len_ns, Tsky_mdl)
    vis += foregrounds.pntsrc_foreground(lsts, fqs, bl_len_ns, nsrcs=200)

    # noise
    tsky = noise.resample_Tsky(fqs,lsts,Tsky_mdl=noise.HERA_Tsky_mdl['xx'])
    t_rx = 150.
#     OMEGA_P = (0.72)*np.ones(1024) # from before; fraction of sky telescope is looking at; normalizes noise
    OMEGA_P = noise.bm_poly_to_omega_p(fqs) # default; from the hera_sim docs
    nos_jy = noise.sky_noise_jy(tsky + t_rx, fqs, lsts, OMEGA_P)
    vis += nos_jy
    
    # crosstalk, gains
    xtalk = sigchain.gen_whitenoise_xtalk(fqs)
#     g = sigchain.gen_gains(fqs, [1,2,3,4], 0) # default 0.1 # leave out for now
    vis = sigchain.apply_xtalk(vis, xtalk)
#     vis = sigchain.apply_gains(vis, g, (1,2))

    # separate new_vis into real and imaginary channels
    real = vis.real
    imag = vis.imag
    new_vis = np.zeros((vis.shape[0], vis.shape[1], 2))
    new_vis[:, :, 0] = real
    new_vis[:, :, 1] = imag

    return vis, new_vis


def generate_vis_plots(n, lsts, fqs, bl_len_ns):
    '''
    Generate n visibility waterfall plots
    '''
    return np.array([generate_one_vis_plot(lsts, fqs, bl_len_ns)[1] for i in range(n)]) # [1] to get separated vis


##############################
##### Generate datasets #####
##############################

def create_dataset(vis_list, mask_list):
    if vis_list.shape[-1] < 2:
        raise Exception("Need to separate visibilities into real and complex channels!")

    data = np.zeros((vis_list.shape[0], vis_list.shape[1], vis_list.shape[2], 3))
    data[:, :, :, :2] = vis_list # copy vis over on the first two channels

    # apply same mask to each visibility plot
    if len(mask_list) == 1:
        mask = mask_list[0]
        for i, v in enumerate(data):
            v[mask == True] = 0
            v[:, :, 2] = np.logical_not(mask) # Set mask as 3rd channel
            # print(np.count_nonzero(train_dataset[0]==0)) # check number of 0's in a given vis (to check if mask worked)
        labels = vis_list 

    # apply random mask to each visibility plot
    else:
        for i, v in enumerate(data):
            mask = random.choice(mask_list)
            v[mask == True] = 0
            v[:, :, :, 2] = np.logical_not(mask) # Set mask as 3rd channel
            # print(np.count_nonzero(train_dataset[0]==0)) # check number of 0's in a given vis (to check if mask worked)
        labels = vis_list 
    
    # Save files
    prefix = f"{int(time.time())}_{len(vis_list)}_examples_{len(mask_list)}_masks"

    np.save(os.path.join("generated", f"{prefix}_dataset.npy", data))
    np.save(os.path.join("generated", f"{prefix}_labels.npy", labels))
    np.save(os.path.join("generated", f"{prefix}_masks.npy", mask_list))
    
    print("Dataset saved.")
    print(f"Data shape: {data.shape}. Labels shape: {labels.shape}")


##############################
##### main function #####
##############################

def main():
    # cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_masks", type=int, help="Number of generated masks")
    parser.add_argument("--n_examples", type=int, help="Number of examples to generate")
    args = parser.parse_args()

    n_masks = args.n_masks
    n_examples = args.n_examples

    # generate masks
    saurabhs_mask, num_jd, num_freq_channels = load_real_mask(os.path.join(os.path.dirname(__file__), "..", "data", "mask_HERA.hdf5"))
    # initially 1500 x 818
    rfi_widths, start_mask, end_mask = get_RFI_spans(saurabhs_mask[1434], isRow=True)
    rfi_heights = get_RFI_spans(saurabhs_mask[:,166])

    # Calculate start freq, end freq, and # channels
    start_freq = (100 + 100*start_mask/num_freq_channels) / 1000
    end_freq = (200 - 100*end_mask/num_freq_channels) / 1000
    num_reduced_channels = num_freq_channels - start_mask - end_mask

    # set up visibility parameters
    lsts = np.linspace(0, 0.5*np.pi, num_jd, endpoint=False) # local sidereal times; start range, stop range, number of snapshots
    # num_jd = 1500 to match the mask; π/2 ~ 6h
    fqs = np.linspace(start_freq, end_freq, num_reduced_channels, endpoint=False) # frequencies in GHz; start freq, end freq, number of channels
    # fqs = np.linspace(.1, .2, 1024, endpoint=False) # original
    bl_len_ns = np.array([30.,0,0]) # ENU coordinates

    # generate vis list and masks
    vis_list = generate_vis_plots(n_examples, lsts, fqs, bl_len_ns)
    mask_list = generate_masks(n_masks, num_jd, num_reduced_channels, rfi_widths, rfi_heights)

    print(type(vis_list))
    print(type(mask_list))

    # create dataset
    create_dataset(vis_list, mask_list)


if __name__ == "__main__":
    main()



