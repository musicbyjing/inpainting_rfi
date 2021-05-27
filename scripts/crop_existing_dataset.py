import numpy as np
import os
import argparse
import sys

from utils import load_dataset, plot_one_vis, plot_one_mask

def crop_data(data, labels, dim, save, file_id):
    '''
    Crop data, labels, and mask into dim x dim squares
    '''
    l, m, n, channels_data = data.shape[0:4] # l is length, m is # rows, n is # cols
    channels_labels = labels.shape[3]

    data_new = np.zeros((l, dim, dim, channels_data))
    labels_new = np.zeros((l, dim, dim, channels_labels))
    assert labels.shape[0] == l, "Unequal lengths of data, labels, and mask list!"

    for i in range(l):
        data_new[i] = data[i, :dim, :dim, :]
        labels_new[i] = labels[i, :dim, :dim, :]

    if save:
        prefix = f"{file_id}_CROPPED_{dim}x{dim}"
        np.save(os.path.join("data", f"{prefix}_dataset.npy"), data_new)
        np.save(os.path.join("data", f"{prefix}_labels.npy"), labels_new)
        print("Modified dataset saved.")

    print(f"Data shape: {data_new.shape}. Labels shape: {labels_new.shape}.")

##############################
##### main function #####
##############################

def main():
    # cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, help="Size of one dimension (return square data)")
    parser.add_argument("--id", type=str, help="ID of data to use for training")
    parser.add_argument("--no-save", default=True, action="store_false", help="Use this flag to run tests without saving generated files to disk")
    args = parser.parse_args()

    dim = args.dim
    file_id = args.id
    save = args.no_save

    # Load data
    data, labels = load_dataset(file_id)
    print("Original data:", data.shape)
    print("Original labels:", labels.shape)
    ############# CHANGE BELOW LINE WHEN USING MORE THAN ONE MASK #############
    # print("MASK", mask.shape)
    # mask_list = np.repeat(mask, data.shape[0], axis=0)

    crop_data(data, labels, dim, save, file_id)
    
    print("cut_existing_dataset.py has completed.")

if __name__ == "__main__":
    main()