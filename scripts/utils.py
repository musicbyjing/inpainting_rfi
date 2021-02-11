import numpy as np
import os
import matplotlib.pyplot as plt
import aipy, uvtools

def load_dataset(file_id):
    ''' Load dataset, consisting of data, labels, and masks '''
    folder = "data"
    data = np.load(os.path.join(folder, f"{file_id}_dataset.npy"))
    labels = np.load(os.path.join(folder, f"{file_id}_labels.npy"))
    mask = np.load(os.path.join(folder,f"{file_id}_masks.npy"))
    # print("DATA SHAPE", data.shape, "LABELS SHAPE", labels.shape)
    return data, labels, mask

def plot_loss(history, file_id):
    ''' Plot learning curves '''
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{file_id}.png")

def plot_one_vis(input_vis, ylim, MX, DRNG, figsize, title, filepath):
    '''
    Plot one visibility waterfall plot and save it
    MX is max value of color scale in the plot
    DRNG = MX - min value of color scale in the plot
    '''
    # consolidate real and imag channels of vis if necessary
    print(input_vis.shape)
    if input_vis.ndim > 2: 
        vis = np.zeros((input_vis.shape[0], input_vis.shape[1]), dtype = 'complex128')
        vis[:, :] = input_vis[:, :, 0] + input_vis[:, :, 1]*1j
    else:
        vis = input_vis

    # plot
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    fig.sca(ax1)
    uvtools.plot.waterfall(vis, mode='log', mx=MX, drng=DRNG)
    plt.suptitle(title)
    plt.grid(False)  
    plt.colorbar(label=r"Amplitude [log$_{10}$(V/Jy)]")
    plt.ylim(0,ylim)

    fig.sca(ax2)
    uvtools.plot.waterfall(vis, mode='phs')
    plt.grid(False)
    plt.colorbar(label="Phase [rad]")
    plt.ylim(0,ylim)
    plt.xlabel("Frequency channel")

    fig.text(0.02, 0.5, 'LST [rad]', ha='center', va='center', rotation='vertical')

    plt.savefig(filepath)

def plot_one_mask(mask, filepath):
    '''Plot one mask and save it'''
    plt.figure(figsize=(12,12))
    plt.grid(False)
    plt.imshow(mask, cmap='inferno_r') # black is RFI
    plt.savefig(filepath)