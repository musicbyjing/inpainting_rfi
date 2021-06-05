import numpy as np
import random
import argparse
import os.path
import h5py
import itertools
import time
import sys

##############################
### Generate custom masks ###
##############################

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