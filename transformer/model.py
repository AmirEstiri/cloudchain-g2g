import time
import numpy as np
import tensorflow as tf
import logging
from sttn import *
from utils import *
logging.getLogger('tensorflow').setLevel(logging.ERROR)

B = 8
H = 10 # Grid height
W = 10 # Grid width
N = H*W # Nodes in grpah
M = 120 # previous time steps
T = 60 # future time steps
TIME_STEP = 1

d_A = 64
d_G = 64

sttn = SpatioTemporalTransformer(d_A, d_G, T)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
train_loss = tf.keras.metrics.Mean(name='train_loss')

@tf.function
def train_step(inp, tar):
    print(inp.shape)
    s_mask, t_mask = create_mask(inp)

    with tf.GradientTape() as tape:
        pred = sttn(inp, s_mask, t_mask)[0]
        loss = loss_function(tar, pred, 'MSE')

    gradients = tape.gradient(loss, sttn.trainable_variables)
    optimizer.apply_gradients(zip(gradients, sttn.trainable_variables))
    train_loss(loss)

EPOCHS = 1
beginning = time.time()
for epoch in range(EPOCHS):
    for b in range(indf_train.shape[0]):
        inp = indf_train[b]
        tar = outdf_train[b]
        train_step(inp, tar)
        if b % 10 == 0:
            print(f'Epoch {epoch + 1} Batch {b} Loss {train_loss.result():.4f}')   
    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f}')
print(f'Total time: {time.time() - beginning:.2f} secs\n')