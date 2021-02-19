from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

from utils import masked_MSE

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