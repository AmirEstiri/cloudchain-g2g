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
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
        output = self.dense(scaled_attention)
        return output, attention_weights



class EncoderFeatureExtraction(tf.keras.layers.Layer):
    def __init__(self, dG):
        super(EncoderFeatureExtraction, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(dG, (1,1))
        self.conv2 = tf.keras.layers.Conv2D(dG, (1,1))

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class DecoderPredictionLayer(tf.keras.layers.Layer):
    def __init__(self, dG):
        super(DecoderPredictionLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2D(dG, (1,1))
        self.dense = tf.keras.layers.Conv2D(3, (1,1))

    def call(self, x):
        x = self.conv(x)
        x = self.dense(x)
        return x

        

class DecoderContextLayer(tf.keras.layers.Layer):
    def __init__(self, M, W, H, dG):
        super(DecoderContextLayer, self).__init__()
        self.M = M
        self.W = W
        self.H = H
        self.dense1 = tf.keras.layers.Dense(M, activation='relu')
        self.dense2 = tf.keras.layers.Dense(W*H)
        self.conv = tf.keras.layers.Conv2D(dG, (1,1))
        
    def call(self, x):
        x = self.dense1(x)
        x = x[:, :, tf.newaxis]
        x = self.dense2(x)
        x = tf.reshape(x, (-1, self.M, self.W, self.H))
        x = x[:, :, :, :, tf.newaxis]
        x = self.conv(x)
        return x



class SpatialLayer(tf.keras.layers.Layer):
    def __init__(self, dA, dG):
        super(SpatialLayer, self).__init__()
        self.att = Attention(dA, dG)
        self.feed_forward_network = point_wise_feed_forward_network(dG)

    def call(self, x):
        s_out, s_att = self.att(x, x, x)
        out = self.feed_forward_network(s_out)
        return out + x, s_att



class TemporalLayer(tf.keras.layers.Layer):
    def __init__(self, dA, dG):
        super(TemporalLayer, self).__init__()
        self.att = Attention(dA, dG)
        self.feed_forward_network = point_wise_feed_forward_network(dG)

    def call(self, x):
        t_out, t_att = self.att(x, x, x)
        out = self.feed_forward_network(t_out)
        return out + x, t_att



class SpatialTemporalBlock(tf.keras.layers.Layer):
    def __init__(self, dA, dG):
        super(SpatialTemporalBlock, self).__init__()
        self.spatial = SpatialLayer(dA, dG)
        self.temporal = TemporalLayer(dA, dG)

    def call(self, x):
        x, s_att = self.spatial(x)
        x, t_att = self.temporal(x)
        return x, s_att, t_att



class EncoderContextLayer(tf.keras.layers.Layer):
    def __init__(self, compr_size):
        super(EncoderContextLayer, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(10, (1,1))
        self.conv2 = tf.keras.layers.Conv2D(20, (1,1))
        self.dense = tf.keras.layers.Dense(compr_size, activation='sigmoid')
        
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = tf.reshape(x, (x.shape[0], -1))
        x = self.dense(x)
        return x



class Encoder(tf.keras.layers.Layer):
    def __init__(self, dA, dG, compr_size):
        super(Encoder, self).__init__()
        self.feat_layer = EncoderFeatureExtraction(dG)
        self.stb = SpatialTemporalBlock(dA, dG)
        self.enc_layer = EncoderContextLayer(compr_size)

    def call(self, x):
        x = self.feat_layer(x)
        x, s_att, t_att = self.stb(x)
        x = self.enc_layer(x)
        return x, s_att, t_att



class Decoder(tf.keras.layers.Layer):
    def __init__(self, dA, dG, M, W, H):
        super(Decoder, self).__init__()
        self.dec_context = DecoderContextLayer(M, W, H, dG)
        self.stb = SpatialTemporalBlock(dA, dG)
        self.dec_layer = DecoderPredictionLayer(dG)

    def call(self, x):
        x = self.dec_context(x)
        x, s_att, t_att = self.stb(x)
        x = self.dec_layer(x)
        return x, s_att, t_att
