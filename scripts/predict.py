import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


def predict(input):
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
    folder = "models"
    return keras.models.load_model(os.path.join(folder, f"{model_name}.npy"), custom_objects={'masked_MSE': masked_MSE})

def get_predictions(X_train):
    # Get a random example from X_train
    i = random.randint(0, len(X_train)-1)
    
    np.save("pred.npy", predict(X_train[i]))
    np.save("og.npy", X_train[i])
    np.save("true.npy", y_train[i])

    # Add unmasked part of original to prediction
    pred[mask == False] = X_train[i][mask == False][:,:2] # (1-mask)*og + mask*pred
    np.save("pred_masked.npy", preds[i])

def main():
    # cmd line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, help="ID of data to use for making predictions")
    args = parser.parse_args()

    file_id = args.file_id
    ########################
    ### NEED TO FIND A WAY TO GET A TRAINING EXAMPLE AND GROUND TRUTH IN HERE!!! ####
