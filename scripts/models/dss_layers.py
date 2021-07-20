import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

class Conv2dDeepSym(keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, use_max=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_max = use_max
        self.conv = keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding=padding, kernel_initializer='glorot_normal')
        self.conv_s = keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding=padding, kernel_initializer='glorot_normal')
        self.bn = keras.layers.BatchNormalization(axis=4)
        self.bns = keras.layers.BatchNormalization(axis=4)
        # weights initialized in conv and conv_s

    def call(self, x):
        b, n, h, w, c = x.shape
        x1 = self.bn(self.conv(K.reshape(x, (n*b, h, w, c))))
        if self.use_max:
            x2 = self.bns(self.conv_s(K.max(x, axis=1, keepdims=False)[0]))
        else:
            x2 = self.bns(self.conv_s(K.sum(x, axis=1, keepdims=False)))
        x2 = K.reshape(x2, (b, 1, h, w, self.out_channels))
        x2 = K.tile(x2, [1, n, 1, 1, 1])
        x2 = K.reshape(x2, (b*n, h, w, self.out_channels))
        x = x1 + x2
        x = K.reshape(x, (b, n, h, w, self.out_channels))
        return x



