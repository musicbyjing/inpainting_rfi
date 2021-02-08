import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import aipy, uvtools
from hera_sim import foregrounds, noise, sigchain, rfi, simulate

def predict(model, input):
    return model.predict(np.array([input, ]))[0] # since model.predict() needs an array

def masked_MSE(y_true, y_pred):
    '''
    MSE, only over masked areas
    '''
    for yt in y_true: # for each example in the batch
        yt = yt[mask == True]
    for yp in y_pred:
        yp = yp[mask == True]
    loss_val = K.mean(K.square(y_pred - y_true))
    return loss_val

def load_model(model_name):
    folder = "."
    return keras.models.load_model(os.path.join(folder, f"{model_name}"), custom_objects={'masked_MSE': masked_MSE})

def load_data(file_id):
    folder = "data"
    data = np.load(os.path.join(folder, f"{file_id}_Xtest.npy"))
    label = np.load(os.path.join(folder, f"{file_id}_ytest.npy"))
    return data, label

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

def get_prediction(model, data, label):
    '''
    Get one prediction from Xtest and ytest
    '''
    # Get a random example from data
    i = random.randint(0, len(data)-1)
    
    folder = "images"
    pred = predict(model, data[i])
    plot_one_vis(pred, 1500, 2.5, 3, (7,7), "Predicted", os.path.join(folder, "pred.png"))
    plot_one_vis(data[i], 1500, 2.5, 3, (7,7), "Original", os.path.join(folder, "og.png"))
    plot_one_vis(label[i], 1500, 2.5, 3, (7,7), "True", os.path.join(folder, "true.png"))
    
    # Add unmasked part of original to prediction
    pred[mask == False] = data[i][mask == False][:,:2] # (1-mask)*og + mask*pred
    plot_one_vis(pred, 1500, 2.5, 3, (7,7), "Predicted, masked", os.path.join(folder, "pred_masked.npy"))


##############################
##### main function #####
##############################

def main():
    # cmd line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, help="ID of data to use for making predictions")
    parser.add_argument("--model-name", type=str, help="Name of the trained model, including file extension")
    args = parser.parse_args()

    file_id = args.id
    model_name = args.model_name
    
    # load parameters
    model = None
    model = load_model(model_name)
    data, label = load_data(file_id)

    # get predictions
    get_prediction(model, data, label)
    print("predict.py completed.")


if __name__ == "__main__":
    main()