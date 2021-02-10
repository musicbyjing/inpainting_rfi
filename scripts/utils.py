import numpy as np
import os
import matplotlib.pyplot as plt

def load_dataset(file_id):
    folder = "data"
    data = np.load(os.path.join(folder, f"{file_id}_dataset.npy"))
    labels = np.load(os.path.join(folder, f"{file_id}_labels.npy"))
    mask = np.load(os.path.join(folder,f"{file_id}_masks.npy"))
    # print("DATA SHAPE", data.shape, "LABELS SHAPE", labels.shape)
    return data, labels, mask

def plot_loss(history, file_id):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{file_id}.png")