import numpy as np
import os
import argparse
import sys
from utils import load_dataset

def cut_data(data, labels, mask_list, time_div, freq_div, save):
    '''
    Cut data, labels, and mask by time_div and freq_div
    '''
    data_new = []
    labels_new = []
    mask_list_new = []
    for d, l, m in zip(data, labels, mask_list):
        pass


    # if save:
    #     prefix = f"{int(time.time())}_{len(vis_list)}_examples_{len(mask_list)}_masks"
    #     np.save(os.path.join("data", f"{prefix}_dataset.npy"), data_new)
    #     np.save(os.path.join("data", f"{prefix}_labels.npy"), labels_new)
    #     np.save(os.path.join("data", f"{prefix}_masks.npy"), mask_list_new)
    #     print("Modified dataset saved.")
    print(f"Data shape: {data.shape}. Labels shape: {labels.shape}")




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
    print("MASK", mask.shape)
    mask_list = np.repeat(mask, data.shape[0], axis=0)
    print("MASK_LIST", mask_list.shape)
    # print(data.shape)
    # print(labels.shape)

    cut_data(data, labels, mask, time_div, freq_div, save)
    
    print("cut_existing_dataset.py has completed.")

if __name__ == "__main__":
    main()