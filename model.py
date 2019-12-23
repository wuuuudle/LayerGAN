import tensorflow as tf

MAX_SEQ_LEN = 500
GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
VOCAB_SIZE = 3590 + 2


def init_model(embedding_dim, hidden_dim, vocab_size, max_seq_len):
    inp = tf.keras.Input(shape=(max_seq_len,))  # batch_size x seq_len
    emb_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True, name='embedding')

    # discriminator
    x = emb_layer(inp)  # batch_size x (seq_len + 1) x embedding_dim
    x = tf.keras.layers.Conv1D(filters=16, kernel_size=10, padding='same', activation='softmax')(
        x)  # batch_size x (seq_len + 1) x 16
    feature = tf.keras.layers.LSTM(hidden_dim)(x)  # batch_size x hidden_dim
    probabilities = tf.keras.layers.Dense(2, activation='softmax')(feature)
    discriminator = tf.keras.Model(inputs=inp, outputs=probabilities, name='discriminator')

    # generator
    emb = emb_layer(inp)
    decoder = tf.keras.layers.GRU(hidden_dim)(emb)  # batch_size x hidden_dim
    con = tf.keras.backend.concatenate([decoder, feature], axis=1)  # batch_size x (hidden_dim+hidden_dim)
    out = tf.keras.layers.Dense(vocab_size, activation='softmax')(con)  # batch_size x vocab_size
    generator = tf.keras.Model(inputs=inp, outputs=out, name='generator')

    return discriminator, generator


if __name__ == '__main__':
    dis, gen = init_model(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN)
    print(dis, gen)
