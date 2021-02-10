import numpy as np
import os
import argparse
import sys
from skimage.util.shape import view_as_blocks

from utils import load_dataset, plot_one_vis

def split_data(data, labels, mask_list, time_div, freq_div, save):
    '''Cut data, labels, and mask by time_div and freq_div'''
    l, m, n, channels_data = data.shape[0:4]
    channels_labels = labels.shape[3]
    channels_mask = 0
    data_new = []
    labels_new = []
    mask_list_new = []
    assert labels.shape[0] == l and mask_list.shape[0] == l, "Unequal lengths of data, labels, and mask list!"

    for i in range(l):
        data_new.append(split(data[i], m//time_div, n//freq_div, channels_data)) # for a 1500 x 818 image, want blocks of size 750, 409, 3
        labels_new.append(split(labels[i], m//time_div, n//freq_div, channels_labels)) 
        mask_list_new.append(split(mask_list[i], m//time_div, n//freq_div, channels_mask)) 
        print(i)

    data_new = np.array(data_new)
    labels_new = np.array(labels_new)
    mask_list_new = np.array(mask_list_new)

    if save:
        prefix = f"{int(time.time())}_{len(vis_list)}_examples_{len(mask_list)}_masks"
        np.save(os.path.join("data", f"{prefix}_dataset.npy"), data_new)
        np.save(os.path.join("data", f"{prefix}_labels.npy"), labels_new)
        np.save(os.path.join("data", f"{prefix}_masks.npy"), mask_list_new)
        print("Modified dataset saved.")

    # print(f"Data shape: {data_new.shape}. Labels shape: {labels_new.shape}. Mask shape: {mask_list_new.shape}")

def split(array, nrows, ncols, c):
    '''
    Split a matrix into contiguous blocks.
    '''
    if c == 0:
        return view_as_blocks(array, block_shape=(nrows, ncols)).reshape(-1, nrows, ncols)
    else:
        return view_as_blocks(array, block_shape=(nrows, ncols, c)).reshape(-1, nrows, ncols, c)

# def save_images(og, split):
#     '''
#     Testing function to ensure blocks are being cut correctly
#     '''
#     folder = "images"
#     plot_one_vis(og, 1500, 2.5, 3, (7,7), "cut image", os.path.join(folder, "no_cut.png"))
#     for i, img in enumerate(split):
#         plot_one_vis(img, 1500, 2.5, 3, (7,7), "cut image", os.path.join(folder, f"cut{i}.png"))


##############################
##### main function #####
##############################

def main():
    # cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--divide-time-by", type=int, help="Divide vertical axis into this many units")
    parser.add_argument("--divide-freq-by", type=int, help="Divide horizontal axis into this many units")
    parser.add_argument("--id", type=str, help="ID of data to use for training")
    parser.add_argument("--no-save", default=True, action="store_false", help="Use this flag to run tests without saving generated files to disk")
    args = parser.parse_args()

    time_div = args.divide_time_by
    freq_div = args.divide_freq_by
    file_id = args.id
    save = args.no_save

    # Load data
    data, labels, mask = load_dataset(file_id)
    ############# CHANGE BELOW LINE WHEN USING MORE THAN ONE MASK #############
    # print("MASK", mask.shape)
    mask_list = np.repeat(mask, data.shape[0], axis=0)
    # print("MASK_LIST", mask_list.shape)
    # print(data.shape)
    # print(labels.shape)

    split_data(data, labels, mask_list, time_div, freq_div, save)
    
    print("cut_existing_dataset.py has completed.")

if __name__ == "__main__":
    main()