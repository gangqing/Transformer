import tensorflow as tf
from tensorflow import keras
import numpy as np


class ScaleDotProductAttention(keras.layers.Layer):
    """"
    点乘的自注意力机制
    """""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        :param input_shape: list (3, batch_size, n_heads, len)
        :return:
        """
        q_shape = input_shape[0]
        k_shape = input_shape[1]
        v_shape = input_shape[2]
        mask_shape = input_shape[3]
        self.d_k = k_shape[-1]
        self.sofmax = keras.layers.Softmax(axis=-1)

    def call(self, inputs, **kwargs):
        """
        :param inputs: (Q, K, V, attn_mask)
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: 由True和False组成的Tensor，[batch_size, n_heads, len_q, len_k]
        :param kwargs:
        :return:
        """
        super(ScaleDotProductAttention, self).call(inputs)
        Q, K, V, attn_mask = inputs
        # Q 与 K 的转置相乘
        scores = tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2])) / np.sqrt(self.d_k)  # [batch_size, n_heads, len_q, len_k]
        # todo 掩码填充
        # scores = tf.where(attn_mask,  -1e9, scores)
        # scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        attn = self.sofmax(dim=-1)(scores)
        context = tf.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


if __name__ == '__main__':
    q = keras.Input([2, 10, 200])  # [-1, 2, 10, 200]
    k = keras.Input([2, 12, 200])  # [-1, 2, 12, 200]
    v = keras.Input([2, 12, 300])  # [-1, 2, 12, 300]
    attn_mask = keras.Input([2, 300, 300])  # [-1, 2, 300, 300]
    context, attn = ScaleDotProductAttention()([q, k, v, attn_mask])
