import numpy as np
import os
import argparse
import sys

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint

from model import *
from utils import load_dataset, plot_loss

mask = ""

# def masked_MSE(mask):
def loss (y_true, y_pred):
    '''
    MSE, only over masked areas
    '''
    for yt in y_true: # for each example in the batch
        yt = yt[mask == True]
    for yp in y_pred:
        yp = yp[mask == True]
    loss_val = K.mean(K.square(y_pred - y_true))
    return loss_val
    # return loss

def build_and_compile_model():
    model = keras.Sequential([
        keras.layers.Conv2D(24, kernel_size=20, activation='relu', padding='same', input_shape=(1500,818,3), kernel_initializer=keras.initializers.GlorotNormal()),
        # keras.layers.MaxPooling2D((2,2)),
        # keras.layers.UpSampling2D((2,2)),
        keras.layers.Conv2D(24, kernel_size=15, activation='relu', padding='same', kernel_initializer=keras.initializers.GlorotNormal()),
        # keras.layers.MaxPooling2D((2,2)),
        # keras.layers.UpSampling2D((2,2)),
        keras.layers.Conv2D(24, kernel_size=15, activation='relu', padding='same', kernel_initializer=keras.initializers.GlorotNormal()),
        # keras.layers.MaxPooling2D((2,2)),
        # keras.layers.UpSampling2D((2,2)),
        keras.layers.Conv2D(24, kernel_size=9, activation='relu', padding='same', kernel_initializer=keras.initializers.GlorotNormal()),
        keras.layers.Conv2D(24, kernel_size=6, activation='relu', padding='same', kernel_initializer=keras.initializers.GlorotNormal()),
        keras.layers.Conv2D(24, kernel_size=6, activation='relu', padding='same', kernel_initializer=keras.initializers.GlorotNormal()),
        # keras.layers.MaxPooling2D((2,2)),
        # keras.layers.UpSampling2D((2,2)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(2)
    ])
    model.compile(loss=masked_MSE, optimizer=tf.keras.optimizers.Adam(0.001), metrics=[masked_MSE])
    return model

def build_and_compile_AlexNet():
    '''
    based on AlexNet from 'https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98'
    '''
    model = keras.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), padding='same', activation='relu', input_shape=(1500,818,3)),
        keras.layers.BatchNormalization(),
        # keras.layers.MaxPool2D(pool_size=(2,2)),
        # keras.layers.UpSampling2D((2,2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5,5), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        # keras.layers.MaxPool2D(pool_size=(2,2)),
        # keras.layers.UpSampling2D((2,2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        # keras.layers.MaxPool2D(pool_size=(2,2)),
        # keras.layers.UpSampling2D((2,2)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2)
    ])
    model.compile(loss=masked_MSE, optimizer=tf.keras.optimizers.Adam(0.001), metrics=[masked_MSE])
    return model


def main():
    # tf stuff
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, help="Maximum number of epochs")
    parser.add_argument("--id", type=str, help="ID of data to use for training")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--model", type=str, help="architecture to select. Current options are: colab, alex")
    parser.add_argument("--no-save-test", default=True, action="store_false", help="Run without saving generated Xtest and ytest sets")
    parser.add_argument("--compile_only", default=False, action="store_true", help="Quit after compiling and printing model")
    args = parser.parse_args()

    max_epochs = args.max_epochs
    file_id = args.id
    save_test = args.no_save_test
    compile_only = args.compile_only
    model_name = args.model
    batch_size=args.batch_size
    # print(save_test)

    # Get model
    model = None
    if model_name == "colab":
        model = build_and_compile_model()
    elif model_name == "alex":
        model = build_and_compile_AlexNet()
    elif model_name == 'unet':
        model = unet(input_size=(740,409,3))
    else:
        raise Exception("Unsupported model. Please try again")
        sys.exit(0)
    print(model.summary())
    if compile_only:
        sys.exit(0)

    # Load data
    global mask
    data, labels, masks = load_dataset(file_id)
    ############# CHANGE BELOW LINE WHEN USING MORE THAN ONE MASK #############
    mask = mask[0]
    X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(data, labels, masks, test_size=0.2, random_state=42)
    if save_test:
        np.save(os.path.join("data", f"{file_id}_Xtest.npy"), X_test)
        np.save(os.path.join("data", f"{file_id}_ytest.npy"), y_test)
    del data
    print("X_TRAIN", X_train.shape, "X_TEST", X_test.shape, "Y_TRAIN", y_train.shape, "Y_TEST", y_test.shape)
    print("MASK", mask.shape)
    
    # Save checkpoints
    filepath=os.path.join("models", f"{model_name}_weights.best.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_masked_MSE', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # Fit model
    history = model.fit(X_train, y_train, validation_split=0.2, verbose=1, batch_size=batch_size, epochs=max_epochs, callbacks=callbacks_list)
    model.save(os.path.join("models", f"{model_name}_{file_id}_model.h5"))
    plot_loss(history, file_id)
    print("train_model.py has completed.")

if __name__ == "__main__":
    main()