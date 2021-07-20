import numpy as np
import tensorflow as tf
from utils import *


class Attention(tf.keras.layers.Layer):
    def __init__(self, d_A, d_G):
        super(Attention, self).__init__()
        self.d_G = d_G
        self.d_A = d_A
        self.wq = tf.keras.layers.Dense(d_A)
        self.wk = tf.keras.layers.Dense(d_A)
        self.wv = tf.keras.layers.Dense(d_G)
        self.dense = tf.keras.layers.Dense(d_G)

    def call(self, q, k, v, mask):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        output = self.dense(scaled_attention)
        return output, attention_weights


class T2V(tf.keras.layers.Layer):  
    def __init__(self, output_dim=None):
        self.output_dim = output_dim
        super(T2V, self).__init__()
        
    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                      shape=(input_shape[1], self.output_dim),
                      initializer='uniform',
                      trainable=True)
        self.P = self.add_weight(name='P',
                      shape=(input_shape[1], self.output_dim),
                      initializer='uniform',
                      trainable=True)
        self.w = self.add_weight(name='w',
                      shape=(input_shape[1], 1),
                      initializer='uniform',
                      trainable=True)
        self.p = self.add_weight(name='p',
                      shape=(input_shape[1], 1),
                      initializer='uniform',
                      trainable=True)
        super(T2V, self).build(input_shape)
        
    def call(self, x):
        original = self.w * x + self.p
        sin_trans = K.sin(K.dot(x, self.W) + self.P)
        return K.concatenate([sin_trans, original], -1)



class FeatureAggregation(tf.keras.layers.Layer):
    def __init__(self, d_G):
        super(FeatureAggregation, self).__init__()
        self.t2v = T2V(20)
        self.conv = tf.keras.layers.Conv2D(d_G, (1,1))

    def call(self, x):
        x = x[:, :, :, tf.newaxis]
        x = self.conv(x)
        return x



class DynamicSpatialLayer(tf.keras.layers.Layer):
    def __init__(self, d_A, d_G):
        super(DynamicSpatialLayer, self).__init__()
        self.att = Attention(d_A, d_G)

    def call(self, x, mask):
        x_ = tf.transpose(x, perm=[0, 2, 1, 3])
        ds_out, ds_att = self.att(x_, x_, x_, mask)
        ds_out = tf.transpose(ds_out, perm=[0, 2, 1, 3])
        return ds_out + x, ds_att



class SpatialLayer(tf.keras.layers.Layer):
    def __init__(self, d_A, d_G):
        super(SpatialLayer, self).__init__()
        self.d_A = d_A
        self.d_G = d_G
        self.ds_layer = DynamicSpatialLayer(d_A, d_G)
        self.feed_forward_network = point_wise_feed_forward_network(d_G)

    def call(self, x, mask):
        ds_out, ds_att = self.ds_layer(x, mask)
        out = self.feed_forward_network(ds_out)
        return out + x, None, ds_att



class TemporalLayer(tf.keras.layers.Layer):
    def __init__(self, d_A, d_G):
        super(TemporalLayer, self).__init__()
        self.att = Attention(d_A, d_G)
        self.feed_forward_network = point_wise_feed_forward_network(d_G)

    def call(self, x, mask):
        t_out, t_att = self.att(x, x, x, mask)
        out = self.feed_forward_network(t_out)
        return out + x, t_att



class PredictionLayer(tf.keras.layers.Layer):
    def __init__(self, T):
        super(PredictionLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(T, activation='sigmoid')
        self.conv1 = tf.keras.layers.Conv2D(10, (1,1))
        self.conv2 = tf.keras.layers.Conv2D(20, (1,1))
        
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = tf.reshape(x, (x.shape[0], x.shape[1], -1))
        x = self.dense(x)
        return x


class SpatioTemporalTransformer(tf.keras.Model):
    def __init__(self, d_A, d_G, T):
        super(SpatioTemporalTransformer, self).__init__()
        self.feature_aggregation = FeatureAggregation(d_G)
        self.spatial_transformer = SpatialLayer(d_A, d_G)
        self.temporal_transformer = TemporalLayer(d_A, d_G)
        self.prediction_layer = PredictionLayer(T)

    def call(self, inp, s_mask, t_mask):
        x = self.feature_aggregation(inp)
        s_out, ss_att, ds_att = self.spatial_transformer(x, s_mask)
        t_out, t_att = self.temporal_transformer(s_out, t_mask)
        out = self.prediction_layer(t_out)
        return out, ss_att, ds_att, t_att
