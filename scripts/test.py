from pyuvdata import UVData
import numpy as np
import matplotlib.pyplot as plt
import argparse
import copy, os, itertools, inspect
from tqdm import tqdm
from utils import load_dataset, normalize, denormalize

def load_data(filename):
    uvd = UVData()
    uvd.read(filename) # By default, it would load all the baselines 
    print(uvd.data_array.shape) # (ntime*nbl, 1, nfreq, npol), so (182868, 1, 1024, 4)
    antpairpols = uvd.get_antpairpols() # all the baselines and polarizations in the file 
    print(f"Loaded data with {len(antpairpols)} antpairpols", flush=True)

    baseline_groups, vec_bin_centers, lengths = uvd.get_redundancies()
    print("BASELINE GROUPS")
    print(baseline_groups, flush=True)
    print("lengths")
    print(lengths, flush=True)

    return uvd, antpairpols

def yeet():
    data = np.load("data_real/544real_samples_512x512.npy")
    np.save("1sample_real.npy", data[0])

def yeet2():
    data = np.load("data/1617565338_550_examples_5_masks_dataset.npy")
    labels = np.load("data/1617565338_550_examples_5_masks_labels.npy")
    print("data", data.shape)
    data2 = np.zeros((10, data.shape[1], data.shape[2], data.shape[3]))
    labels2 = np.zeros((10, labels.shape[1], labels.shape[2], labels.shape[3]))
    data2 = data[:10, :, :, :]
    labels2 = labels[:10, :, :, :]
    np.save("1617565338_10_examples_5_masks_dataset.npy", data2)
    np.save("1617565338_10_examples_5_masks_labels.npy", labels2)

def test_norm():
    data, labels, _ = load_dataset("1616898929_544_examples_5_masks")
    data_norm, labels_norm, mean, std = normalize(data, labels)
    data_denorm, labels_denorm = denormalize(data_norm, labels_norm, mean, std)
    print(np.all(data == data_denorm))

def main():
    print("starting", flush=True)
    # filename = os.path.join('data_real', 'sample.uvh5')
    # uvd, antpairpols = load_data(filename)
    yeet2()

    print("test.py complete.", flush=True)

        
if __name__ == "__main__":
    main()