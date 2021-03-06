import numpy as np
import os
import matplotlib.pyplot as plt
import aipy
import uvtools

def normalize_over_all(data):
    '''
    Normalize over non-masked areas by subtracting mean of all pixels of ALL IMAGES and 
    dividing by std
    '''
    nonzero = data != 0
    mean = np.mean(data[nonzero])
    std = np.std(data[nonzero])
    data -= mean
    data /= std
    return data

def denormalize_over_all(data, labels, mean, std):
    '''
    De-normalize
    '''
    data += mean
    data *= std
    return data

def normalize_over_each(data):
    '''
    Normalize over non-masked areas by subtracting mean of all pixels in EACH IMAGE and 
    dividing by std
    '''
    for d in data:
        nonzero = d != 0
        mean = np.mean(d[nonzero])
        std = np.std(d[nonzero])
        d -= mean
        d /= std
    return data

def load_dataset(file_id):
    ''' Load dataset, consisting of data and labels '''
    folder = "data"
    data = np.load(os.path.join(folder, f"{file_id}_dataset.npy"))
    labels = np.load(os.path.join(folder, f"{file_id}_labels.npy"))
    # print("DATA SHAPE", data.shape, "LABELS SHAPE", labels.shape)
    return data, labels

def plot_loss(history, file_id):
    ''' Plot learning curves '''
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{file_id}.png")

def plot_one_vis(input_vis, MX, DRNG, figsize, title, filepath, show_pred_area=False):
    # Removed ylim (and xlim)
    '''
    Plot one visibility waterfall plot and save it
    MX is max value of color scale in the plot
    DRNG = MX - min value of color scale in the plot
    when show_pred_area = True, the prediction area is highlighted in a different color (which is 
        only useful for visualization in original images)
    '''
    # consolidate real and imag channels of vis if necessary
    print("Input visibility shape:", input_vis.shape)
    if input_vis.ndim > 2:
        vis = np.zeros(
            (input_vis.shape[0], input_vis.shape[1]), dtype='complex128')
        vis[:, :] = input_vis[:, :, 0] + input_vis[:, :, 1]*1j
    else:
        vis = input_vis

    # get sim - real mask
    if show_pred_area == True:
        prediction_area = input_vis[:, :, 3]
        masked = np.ma.masked_where(prediction_area == 0, prediction_area)

    # plot
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    fig.sca(ax1)
    plt.suptitle(title)
    if show_pred_area == True:
        plt.imshow(masked, alpha=1, cmap='Wistia_r', interpolation='none',
                   aspect='auto')  # plot real mask in different color
    uvtools.plot.waterfall(vis, mode='log') #, mx=MX, drng=DRNG)
    plt.grid(False)
    plt.colorbar(label=r"Amplitude [log$_{10}$(V/Jy)]")
    # plt.xlim(0,xlim)
    # plt.ylim(0,ylim)

    fig.sca(ax2)
    uvtools.plot.waterfall(vis, mode='phs', cmap='twilight')
    plt.colorbar(label="Phase [rad]")
    if show_pred_area == True:
        plt.imshow(masked, alpha=1, cmap='Wistia_r',
                   interpolation='none', aspect='auto')
    plt.grid(False)
    # plt.xlim(0,xlim)
    # plt.ylim(0,ylim)
    plt.xlabel("Frequency channel")

    fig.text(0.02, 0.5, 'LST [rad]', ha='center',
             va='center', rotation='vertical')

    plt.savefig(filepath)

def plot_one_mask(mask, filepath):
    '''Plot one mask and save it'''
    plt.figure(figsize=(12, 12))
    plt.grid(False)
    plt.imshow(mask, cmap='inferno_r')  # black is RFI
    plt.savefig(filepath)

def plot_history_csv(input_filepath, output_filepath):
    '''Plot learning curves using a csv log file'''
    data = np.genfromtxt(input_filepath, delimiter=",", names=[
                         "epoch", "loss", "loss_fn", "val_loss", "val_loss_fn"])
    plt.figure(figsize=(12, 12))
    plt.yscale("log")
    plt.plot(data["epoch"], data["loss"], label="loss")
    plt.plot(data["epoch"], data["loss_fn"], label="loss_fn")
    plt.plot(data["epoch"], data["val_loss"], label="val_loss")
    plt.plot(data["epoch"], data["val_loss_fn"], label="val_loss_fn")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(output_filepath)
