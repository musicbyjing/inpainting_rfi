import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def load_dataset(id):
    folder = "data"
    data = np.load(os.path.join(folder, f"{id}_dataset.npy"))
    labels = np.load(os.path.join(folder, f"{id}_labels.npy"))
    mask = np.load(os.path.join(folder,f"{id}_mask.npy"))
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
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
#         keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2)
    ])
    
    model.compile(loss=masked_MSE, optimizer=tf.keras.optimizers.Adam(0.001), metrics=[masked_MSE])
    return model


def plot_loss(history, id):
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
    args = parser.parse_args()

    max_epochs = args.max_epochs
    file_id = args.id

    # data, labels, mask = load_dataset(file_id)
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)
    # del data
    # print("X_TRAIN", X_train.shape, "X_TEST", X_test.shape, "Y_TRAIN", y_train.shape, "Y_TEST", y_test.shape)
    # print("MASK", mask.shape)

    model = None
    model = build_and_compile_model()
    model.summary()

    history = model.fit(X_train, y_train, validation_split=0.2, verbose=1, epochs=max_epochs)
    model.save('model.h5')

if __name__ == "__main__":
    main()



plot_loss(history)


# In[95]:


files.download('model.h5')


# In[96]:


while True:pass


# ## Make predictions

# In[ ]:


model.layers[0].get_weights()


# In[ ]:


# model = keras.models.load_model('model.h5', custom_objects={'masked_MSE': masked_MSE})


# In[97]:


def make_predictions(input):
  preds = model.predict(input)
  # print(preds)
  return preds

def download_predictions(preds, i):
  np.save("pred.npy", preds[i])
  np.save("og.npy", X_train[i])
  np.save("true.npy", y_train[i])
  files.download('pred.npy')
  files.download('og.npy')
  files.download('true.npy')

  # Apply mask to predictions
  # for j, pred in enumerate(preds):
    # pred[mask == False] = X_train[j][mask == False][:,:2] # (1-mask)*og + mask*pred
  
  # np.save("pred_masked.npy", preds[i])
  # files.download('pred_masked.npy')


# In[ ]:


# eye = np.dstack([np.zeros((1500, 818))]*2)
# eye = np.dstack([np.eye(1500, 818)]*2)
# print(eye.shape)
# pred = model.predict(np.array([eye,]))
np.save("pred.npy", X_train[6])
files.download('pred.npy')


# In[ ]:


pred.shape


# In[98]:


preds = make_predictions(X_train)
# print(X_train[0].shape)
# print(mask.shape)


# In[99]:


download_predictions(preds, 6)


# In[ ]:


plt.imshow(preds[0][:,:,0])


# In[ ]:




