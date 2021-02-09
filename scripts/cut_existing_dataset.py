import numpy as np
import os
import argparse
import sys
from utils import load_dataset

def main():
    # cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--divide-time-by", type=int, help="Divide vertical axis into this many units")
    parser.add_argument("--divide-freq-by", type=int, help="Divide horizontal axis into this many units")
    parser.add_argument("--id", type=str, help="ID of data to use for training")
    args = parser.parse_args()

    time_div = args.divide_time_by
    freq_div = args.divide_freq_by
    file_id = args.id

    # Load data
    data, labels, mask = load_dataset(file_id)
    ############# CHANGE BELOW LINE WHEN USING MORE THAN ONE MASK #############
    mask = mask[0]
    print("X_TRAIN", X_train.shape, "X_TEST", X_test.shape, "Y_TRAIN", y_train.shape, "Y_TEST", y_test.shape)
    print("MASK", mask.shape)
    
    print("cut_existing_dataset.py has completed.")

if __name__ == "__main__":
    main()