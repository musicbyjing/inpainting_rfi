import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from utils import plot_one_vis, masked_MSE, masked_MSE_multiple_masks

### Currently, chooses a random image from X_test and makes a prediction. 
### Saves the original, predicted, and ground truth as png's.

mask = ""

def load_model(model_name, mask=None):
    folder = "models"
    return keras.models.load_model(os.path.join(folder, f"{model_name}"), compile=False)
    # custom_objects={'loss_fn': masked_MSE_multiple_masks})

def load_data(file_id):
    folder = "data"
    data = np.load(os.path.join(folder, f"{file_id}_dataset.npy"))
    labels = np.load(os.path.join(folder, f"{file_id}_labels.npy"))
    return data, labels

def load_real_data(file_id):
    folder = "data_real"
    data = np.load((os.path.join(folder, f"{file_id}.npy")))
    return data

def predict(model, input):
    return model.predict(np.array([input, ]))[0] # since model.predict() needs an array

def get_prediction(model, data, labels, model_name, ground_truth):
    '''
    Get one prediction from Xtest and ytest
    '''
    # Get a random example from data
    i = random.randint(0, len(data)-1)
    
    folder = "images"
    plot_one_vis(data[i], 2.5, 3, (7,7), "Original", os.path.join(folder, f"{model_name}_og.png"), show_pred_area=True)
    pred = predict(model, data[i])
    plot_one_vis(pred, 2.5, 3, (7,7), "Predicted", os.path.join(folder, f"{model_name}_pred.png"))
    
    if not ground_truth:
        plot_one_vis(labels[i], 2.5, 3, (7,7), "True", os.path.join(folder, f"{model_name}_true.png"))
    
        # Add unmasked part of original to prediction
        # mask = masks[i]
        # print("PRED", pred.shape)
        # print("MASK", mask.shape)
        # Something's wrong with the dimensions in the below line...
        # pred[:, :][mask == False] = data[i][:, :, :2][mask == False] # (1-mask)*og + mask*pred 
        # plot_one_vis(pred, 2.5, 3, (7,7), "Predicted, masked", os.path.join(folder, f"{model_name}_pred_masked.png"))


##############################
##### main function #####
##############################

def main():
    # cmd line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, help="ID of data to use for making predictions")
    parser.add_argument("--model-name", type=str, help="Name of the trained model, including file extension")
    parser.add_argument("--no-ground-truth", default=False, action="store_true", help="Predict on real data (without ground truth)")
    args = parser.parse_args()

    file_id = args.id
    model_name = args.model_name
    ground_truth = args.no_ground_truth
    
    model = None
    model = load_model(model_name)

    # load parameters
    if ground_truth:
        data = load_real_data(file_id)
        get_prediction(model, data, None, None, model_name, ground_truth)
    else:
        data, labels = load_data(file_id)
        get_prediction(model, data, labels, model_name, ground_truth)

    # get predictions
    
    print("predict.py completed.")

if __name__ == "__main__":
    main()