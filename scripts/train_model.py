import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

mask = ""

def load_dataset(file_id):
    folder = "data"
    data = np.load(os.path.join(folder, f"{file_id}_dataset.npy"))
    labels = np.load(os.path.join(folder, f"{file_id}_labels.npy"))
    mask = np.load(os.path.join(folder,f"{file_id}_masks.npy"))
    # print("DATA SHAPE", data.shape, "LABELS SHAPE", labels.shape)
    return data, labels, mask

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

def build_and_compile_model():
    '''
    based on AlexNet from 'https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98'
    '''
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), padding='same', activation='relu', input_shape=(1500,818,3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(2,2)),
        keras.layers.UpSampling2D((2,2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5,5), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(2,2)),
        keras.layers.UpSampling2D((2,2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(2,2)),
        keras.layers.UpSampling2D((2,2)),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2)
    ])
    
    model.compile(loss=masked_MSE, optimizer=tf.keras.optimizers.Adam(0.001), metrics=[masked_MSE])
    return model

def plot_loss(history, file_id):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
  plt.savefig(f"{file_id}.png")


def main():
    # cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, help="Maximum number of epochs")
    parser.add_argument("--id", type=str, help="ID of data to use for training")
    parser.add_argument("--no-save-test", default=True, action="store_false", help="Run without saving generated Xtest and ytest sets")
    args = parser.parse_args()

    max_epochs = args.max_epochs
    file_id = args.id
    save_test = args.no_save_test

    data, labels, mask = load_dataset(file_id)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
    if save_test:
        np.save(os.path.join("data", f"{file_id}_Xtest.npy"), X_test)
        np.save(os.path.join("data", f"{file_id}_ytest.npy"), y_test)
    del data
    print("X_TRAIN", X_train.shape, "X_TEST", X_test.shape, "Y_TRAIN", y_train.shape, "Y_TEST", y_test.shape)
    print("MASK", mask.shape)

    model = None
    model = build_and_compile_model()
    print(model.summary())

    history = model.fit(X_train, y_train, validation_split=0.2, verbose=1, epochs=max_epochs)
    # model.save('model.h5')
    ###############################
    # NEED TO GET MASK INTO MSE
    plot_loss(history, file_id)

if __name__ == "__main__":
    main()