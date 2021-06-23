from pyuvdata import UVData
import numpy as np
import matplotlib.pyplot as plt
import argparse
import copy, os, itertools, inspect
from tqdm import tqdm
from utils import load_dataset, plot_history_csv
from crop_existing_dataset import crop_data

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

def plot_history():
    input = 'logs/unet_1617565338_550_examples_5_masks_train_log.csv'
    plot_history_csv(input, "with_norm.png")

'''
Extract one sample from a dataset and save it to the same place"
'''
def extract_one_sample(filepath):
    data = np.load(filepath)
    prefix, ext = os.path.splitext(filepath)
    np.save(f"{prefix}_1sample{ext}", data[0])

def extract_masks():
    data = np.load("data/1616898929_544_examples_5_masks_dataset.npy")
    masks = np.zeros((data.shape[0], data.shape[1], data.shape[2], 1))
    masks = data[:, :, :, 2]
    np.save("masks_from_1616898929_544_examples_5_masks.npy", masks)
    print(masks.shape)

'''
Given an array of real samples, extract masks by getting the areas where data == 0
'''
def extract_masks_real():
    data = np.load("data_real/544real_samples_512x512.npy")
    masks = np.array([d[:,:,2] for d in data])
    np.save("masks_real.npy", masks) # True -> masked
    np.save("masks_real_1sample.npy", masks[0])

'''
Apply masks to data
'''
def apply_masks_to_data(data, masks):
    # res = np.array([data[mask == True] = 0 for data, mask in zip(data, masks)]) 
    res = []
    for d, m in zip(data, masks):
        print(d.shape)
        print(m.shape)
        d[:,:,:2][m == True] = 0
        d[:,:,2][m == True] = 1
        d[:,:,2][m == False] = 0
        res.append(d)
    print(len(res))
    res = np.array(res)
    np.save("vis_list_sim_544_with_applied_real_masks.npy", res)
    np.save("vis_list_sim_544_with_applied_real_masks_1sample.npy", res[0])

'''
Plot scatter of labels vs. predictions
'''
def scatter(label, pred):
    pred_area = label[:, :, 3].astype(int) # prediction area
    x = label[:,:,:2][pred_area]
    y = pred[pred_area]
    print(x.shape, y.shape)
    plt.scatter(x.real.flatten()[::5], y.real.flatten()[::5], alpha=0.1)
    plt.savefig(os.path.join("images", f"real_scatter.png"))
    plt.scatter(x.imag.flatten()[::5], y.imag.flatten()[::5], alpha=0.1)
    plt.savefig(os.path.join("images", f"imag_scatter.png"))


def test_norm():
    data, labels, _ = load_dataset("1616898929_544_examples_5_masks")
    data_norm, labels_norm, mean, std = normalize(data, labels)
    data_denorm, labels_denorm = denormalize(data_norm, labels_norm, mean, std)
    print(np.all(data == data_denorm))

def crop():
    data = np.load("vis_list.npy")
    crop_data(data, data, 512, True, "sim_vis_no_rfi")

def main():
    print("starting", flush=True)
    # filename = os.path.join('data_real', 'sample.uvh5')
    # uvd, antpairpols = load_data(filename)
    scatter(np.load("images/unet_1623596629_555_examples_5_masks_trx1500_weights_best.hdf5_true.npy"), np.load("images/unet_1623596629_555_examples_5_masks_trx1500_weights_best.hdf5_pred.npy"))

    print("test.py complete.", flush=True)

        
if __name__ == "__main__":
    main()