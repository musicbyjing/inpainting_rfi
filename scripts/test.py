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
    print(f"Loaded {filename} with {len(antpairpols)} antpairpols", flush=True)

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

def make_rimez_trainable():
    data = np.load("data/not_masked_RIMEz_dataset.npy")
    data_new = np.zeros((*data.shape[:3], 3))
    print(data.shape)
    for i in range(len(data)):
        print(data[i].shape)
        reverse = np.logical_not(data[i][:, :, 3]) # reverse mask
        data_new[i][:, :, :2] = data[i][:, :, :2]
        data_new[i][:, :, 2] = reverse
    np.save("RIMEz_vis.npy", data_new)
    np.save("RIMEz_vis_1sample.npy", data_new[14])
    print(data.shape)

def crop_center(img,cropx,cropy):
    y,x,_ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def yeet():
    data = np.load("data/1624637389_216_examples_5_masks_MOD_dataset.npy")
    labels = np.load("data/1624637389_216_examples_5_masks_MOD_labels.npy")

    d = []
    d.append(data[:5])
    d.append(data[5:10])
    d.append(data[10:15])
    l = []
    l.append(labels[:5])
    l.append(labels[5:10])
    l.append(labels[10:15])

    np.save("color_matching_exp/dataset.npy", np.array(d))
    np.save("color_matching_exp/labels.npy", np.array(l))

def plot_history():
    plot_history_csv("logs/unet_1624637389_216_examples_5_masks_MOD_train_log.csv", "images/unet_1624637389_216_examples_5_masks_MOD_curves.png")

def yeet2():
    data = np.load("color_matching_exp/dataset.npy")
    labels = np.load("color_matching_exp/labels.npy")
    data2 = []
    labels2 = []
    for d, l in zip(data, labels):
        d2 = []
        for im in d:
            d2.append(crop_center(im, 64, 64))
        data2.append(d2)
        l2 = []
        for im in l:
            l2.append(crop_center(im, 64, 64))
        labels2.append(l2)
    print(np.array(data2).shape)
    np.save("color_matching_exp/dataset_64x64.npy", np.array(data2))
    np.save("color_matching_exp/labels_64x64.npy", np.array(labels2))


def yeet3():
    sets = np.load("sets/(0, 1, 'ee')_unshifted.npy")
    ref = sets[0]
    s = 623
    image_fft = np.pad(sets[1], [(s,0), (0, 0)], mode='constant')[:-s, :]
    s2 = 113
    image_conv = np.pad(sets[1], [(s2,0), (0, 0)], mode='constant')[:-s2, :]

    frac_err1 = frac_error(image_fft, ref)
    frac_err2 = frac_error(image_conv, ref)

    plt.figure(figsize=(10,10))
    plt.hist(frac_err1, density=True, bins=40000)
    plt.savefig("images/fft_error_total.png")
    plt.figure(figsize=(10,10))
    plt.hist(frac_err2, density=True, bins=40000)
    plt.savefig("images/conv_error_total.png")


def frac_error(im, ref):
    res = []
    for i, r in zip(im, ref):
        res.append(np.abs(i-r)/r)
    return res

plt.legend()



def main():
    print("starting", flush=True)
    yeet3()
    print("test.py complete.", flush=True)

        
if __name__ == "__main__":
    main()