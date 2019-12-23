import tensorflow as tf
from dataLoader import DataLoader
from model import init_model
import numpy as np

MAX_SEQ_LEN = 500
GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
VOCAB_SIZE = 3590 + 2


def ppx(y_true, y_pred):
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred)
    perplex = tf.keras.backend.cast(tf.keras.backend.pow(math.e, tf.keras.backend.mean(loss, axis=-1)),
                                    tf.keras.backend.floatx())
    return perplex


data = DataLoader(VOCAB_SIZE, MAX_SEQ_LEN, 'text.txt')
x, y = data.read_gen_all()
dis, gen = init_model(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN)
# pre-train gen
gen.compile(loss=ppx, optimizer='adam', metrics=['accuracy'])
gen.fit(x, y, batch_size=512, epochs=500, validation_split=0.1, callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2, verbose=2, mode='min'),
    tf.keras.callbacks.CSVLogger('pre-gen.csv'),
    tf.keras.callbacks.ModelCheckpoint('pre-gen.h5', monitor='loss', save_best_only=True, mode='min')
])
# pre-train gen

# pre-train dis
dis.get_layer('embedding').trainable = False  # Freeze embedding
true, fake = data.read_dis_data(gen)
X = np.concatenate([true, fake])
Y = np.zeros([X.shape[0], 2])
Y[0:true.shape[0], 1] = 1
Y[fake.shape[0]:, 0] = 1
dis.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
dis.fit(X, Y, epochs=1, shuffle=True, batch_size=32, callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1, verbose=2, mode='min'),
    tf.keras.callbacks.CSVLogger('pre-dis.csv'),
    tf.keras.callbacks.ModelCheckpoint('pre-dis.h5', monitor='loss', save_best_only=True, mode='min')
])
# pre-train dis

# train_all
gen.get_layer('embedding').trainable = True
x, reward = data.rollout_reward(gen, dis, 2)
gen.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
gen.fit(x, y, epochs=1, batch_size=256, sample_weight=reward, callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2, verbose=2, mode='min'),
    tf.keras.callbacks.CSVLogger('trainALL.csv'),
    tf.keras.callbacks.ModelCheckpoint('trainALL.h5', monitor='loss', save_best_only=True, mode='min')
])
# train_all
