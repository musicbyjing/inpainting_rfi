import numpy as np
import os
import argparse
import sys

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from models_unet import *
from models_deep import *
from utils import load_dataset, plot_loss, normalize

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
    parser.add_argument("--normalize", default=False, action="store_true", help="Normalize data before sending into NN")
    parser.add_argument("--trim_all", default=False, action="store_true", help="trim data, labels, masks to 256x256 squares")
    parser.add_argument("--no-save-test", default=True, action="store_false", help="Run without saving generated Xtest and ytest sets")
    parser.add_argument("--compile_only", default=False, action="store_true", help="Quit after compiling and printing model")
    args = parser.parse_args()

    max_epochs = args.max_epochs
    file_id = args.id
    batch_size=args.batch_size
    model_name = args.model
    norm = args.normalize
    trim_all = args.trim_all
    save_test = args.no_save_test
    compile_only = args.compile_only
    # print(save_test)

    print(np.version.version)

    # Load data
    data, labels = load_dataset(file_id)
    if norm:
        data = normalize(data)

    # Get model
    model = None
    if model_name == "colab":
        model = build_and_compile_model()
    elif model_name == "alex":
        model = build_and_compile_AlexNet()
    elif model_name == 'unet':
        model = unet(input_size=(512,512,4))
    else:
        raise Exception("Unsupported model. Please try again")
        sys.exit(0)
    print(model.summary())
    if compile_only:
        sys.exit(0)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    if save_test:
        np.save(os.path.join("data", f"{file_id}_Xtest.npy"), X_test)
        np.save(os.path.join("data", f"{file_id}_ytest.npy"), y_test)
    del data
    print("X_TRAIN", X_train.shape, "X_TEST", X_test.shape, "Y_TRAIN", y_train.shape, "Y_TEST", y_test.shape)
    
    # Calllbacks
    filepath=os.path.join("models", f"{model_name}_{file_id}_weights_best.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    csv_logger = CSVLogger(os.path.join("logs", f"{model_name}_{file_id}_train_log.csv"), append=False)
    callbacks_list = [checkpoint, csv_logger]

    # Fit model
    history = model.fit(X_train, y_train, validation_split=0.2, verbose=1, batch_size=batch_size, epochs=max_epochs, callbacks=callbacks_list)
    model.save(os.path.join("models", f"{model_name}_{file_id}_ending_weights.h5"))
    plot_loss(history, file_id)
    print("train_model.py has completed.")

if __name__ == "__main__":
    main()