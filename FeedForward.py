import tensorflow as tf


def feed_forward_network(d_model, diff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(diff, use_bias=False, activation='relu'),
        tf.keras.layers.Dense(d_model, use_bias=False)  # use_bias必须是False，不然对于全是0的词向量也会起到作用
    ])