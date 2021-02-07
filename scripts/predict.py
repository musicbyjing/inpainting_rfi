import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


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