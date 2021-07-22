import numpy as np
import tensorflow as tf
from models import *


def scaled_dot_product_attention(q, k, v):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dA = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dA)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output


def loss_function(real, pred):
    loss = tf.square(tf.cast(real, dtype=tf.float32) - tf.cast(pred, dtype=tf.float32))
    return tf.reduce_mean(loss)


def point_wise_feed_forward_network(d_G):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d_G, activation='relu'),
    ])