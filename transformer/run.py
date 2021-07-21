import numpy as np
import tensorflow as tf

from models import *
from utils import *

#################################################

EPOCHS = 10
B = 16
W, H = 64, 64
M = 15
dA, dG = 128, 256
compr_size = 200

#################################################

enc = Encoder(dA, dG, compr_size)
dec = Decoder(dA, dG, M, W, H)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
train_loss = tf.keras.metrics.Mean(name='train_loss')

#################################################

@tf.function
def train_step(inp):
    with tf.GradientTape() as tape:
        context = enc(inp)[0]
        pred = dec(context)[0]
        loss = loss_function(inp, pred)
    gradients = tape.gradient(loss, enc.trainable_variables+dec.trainable_variables)
    optimizer.apply_gradients(zip(gradients, enc.trainable_variables+dec.trainable_variables))
    train_loss(loss)

#################################################

indf_train, outdf_train = prepare_video_data_train('data/aff-wild/')
for epoch in range(EPOCHS):
    for b in range(indf_train.shape[0]):
        inp = indf_train[b]
        train_step(inp)
        if b % 10 == 0:
            print(f'Epoch {epoch + 1} Batch {b} Loss {train_loss.result():.4f}')   
    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f}')

#################################################