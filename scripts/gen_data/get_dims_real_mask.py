import numpy as np
import random
import argparse
import os.path
import h5py
import itertools
import time
import sys


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


def generate_masks_prep():
    '''
    Do calculations to get RFI spans
    '''
    # generate masks
    saurabhs_mask, num_jd, num_freq_channels = load_real_mask(os.path.join("data", "mask_HERA.hdf5"))
    # initially 1500 x 818
    rfi_widths, start_mask, end_mask = get_RFI_spans(saurabhs_mask[1434], isRow=True)
    rfi_heights = get_RFI_spans(saurabhs_mask[:,166])

    # Calculate start freq, end freq, and # channels
    start_freq = (100 + 100*start_mask/num_freq_channels) / 1000
    end_freq = (200 - 100*end_mask/num_freq_channels) / 1000
    num_reduced_channels = num_freq_channels - start_mask - end_mask

    return num_jd, num_reduced_channels, start_freq, end_freq, rfi_widths, rfi_heights