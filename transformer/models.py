import numpy as np
import tensorflow as tf
from utils import *


class Attention(tf.keras.layers.Layer):
    def __init__(self, dA, dG):
        super(Attention, self).__init__()
        self.dG = dG
        self.dA = dA
        self.wq = tf.keras.layers.Dense(dA)
        self.wk = tf.keras.layers.Dense(dA)
        self.wv = tf.keras.layers.Dense(dG)
        self.dense = tf.keras.layers.Dense(dG)

    def call(self, q, k, v):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        scaled_attention = scaled_dot_product_attention(q, k, v)
        output = self.dense(scaled_attention)
        return output


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, dA, dG):
        super(AttentionBlock, self).__init__()
        self.att = Attention(dA, dG)
        self.feed_forward_network = point_wise_feed_forward_network(dG)

    def call(self, x):
        out = self.att(x, x, x)
        out = self.feed_forward_network(out)
        return out


class AEConvolutionalEncoder(tf.keras.Model):
    def __init__(self):
        super(AEConvolutionalEncoder).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, (5,5), padding='same', strides=2, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(32, (5,5), padding='same', strides=2, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(32, (5,5), padding='same', strides=1, activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(32, (5,5), padding='same', strides=1, activation='relu')
        self.conv5 = tf.keras.layers.Conv2D(10, (5,5), padding='same', strides=1, activation='relu')
        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)


class AEConvolutionalDecoder(tf.keras.Model):
    def __init__(self, M, W, H):
        super(AEConvolutionalDecoder).__init__()
        self.reshape1 = tf.keras.layers.Reshape((56, 56, -1))
        self.conv1t = tf.keras.layers.Conv2DTranspose(32, (5,5), padding='same', strides=1, activation='relu')
        self.conv2t = tf.keras.layers.Conv2DTranspose(32, (5,5), padding='same', strides=1, activation='relu')
        self.conv3t = tf.keras.layers.Conv2DTranspose(32, (5,5), padding='same', strides=1, activation='relu')
        self.conv4t = tf.keras.layers.Conv2DTranspose(16, (5,5), padding='same', strides=2, activation='relu')
        self.conv5t = tf.keras.layers.Conv2DTranspose(3*M, (5,5), padding='same', strides=2, activation='sigmoid')
        self.reshape2 = tf.keras.layers.Reshape((M, W, H, 3))

    def call(self, x):
        x = self.reshape1(x)
        x = self.conv1t(x)
        x = self.conv1t(x)
        x = self.conv1t(x)
        x = self.conv1t(x)
        x = self.reshape2(x)
