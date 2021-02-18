import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from utils import plot_one_vis

### Currently, chooses a random image from X_test and makes a prediction. 
### Saves the original, predicted, and ground truth as png's.

mask = ""

def load_model(model_name):
    folder = "models"
    return keras.models.load_model(os.path.join(folder, f"{model_name}"), custom_objects={'masked_MSE': masked_MSE})

def load_data(file_id):
    folder = "data"
    data = np.load(os.path.join(folder, f"{file_id}_Xtest.npy"))
    label = np.load(os.path.join(folder, f"{file_id}_ytest.npy"))
    return data, label
    
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

def get_prediction(model, data, label, model_name):
    '''
    Get one prediction from Xtest and ytest
    '''
    # Get a random example from data
    i = random.randint(0, len(data)-1)
    
    folder = "images"
    pred = predict(model, data[i])
    plot_one_vis(pred, 1500, 2.5, 3, (7,7), "Predicted", os.path.join(folder, f"{model_name}_pred.png"))
    plot_one_vis(data[i], 1500, 2.5, 3, (7,7), "Original", os.path.join(folder, f"{model_name}_og.png"))
    plot_one_vis(label[i], 1500, 2.5, 3, (7,7), "True", os.path.join(folder, f"{model_name}_true.png"))
    
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
    get_prediction(model, data, label, model_name)
    print("predict.py completed.")

if __name__ == "__main__":
    main()