import numpy as np
import tensorflow as tf
from sttn import *


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    d_A = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(d_A)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


def create_mask(seq):
    x = tf.cast(tf.math.equal(seq, -1.0), tf.float32)
    x_ = tf.transpose(x, perm=[0, 2, 1])
    t_mask = tf.matmul(x[:, :, :, np.newaxis], x[:, :, np.newaxis, :])
    s_mask = tf.matmul(x_[:, :, :, np.newaxis], x_[:, :, np.newaxis, :])
    return s_mask, t_mask


def loss_function(real, pred, loss_func):
    mask = tf.math.logical_not(tf.math.equal(real, -1.0))
    if loss_func == 'MSE':
        loss = tf.square(tf.cast(real, dtype=tf.float32) - tf.cast(pred, dtype=tf.float32))
    elif loss_func == 'MAE':
        loss = tf.abs(tf.cast(real, dtype=tf.float32) - tf.cast(pred, dtype=tf.float32))
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


def point_wise_feed_forward_network(d_G):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d_G, activation='relu'),
        # tf.keras.layers.Dense(d_G, activation='relu'),
        # tf.keras.layers.Dense(d_G, activation='relu')
    ])