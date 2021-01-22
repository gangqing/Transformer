import Utils
from Config import Config
from tensorflow import keras
import tensorflow as tf
import numpy as np
from MultiHeadAttention import MultiHeadAttention
from FeedForward import feed_forward_network


class Encoder(keras.layers.Layer):
    def __init__(self, cfg: Config):
        super(Encoder, self).__init__()
        self.cfg = cfg
        # 编码层, 对每个词进行编码, 编码后的每个字的向量长度是d_model；
        # 如果使用one-hot编码，词向量的维度是8500，这是一个非常稀疏的矩阵，但使用embedding得到的词向量将压缩在128维，
        # 一方面可以大大地提高模型速度，另一方面可以得到不同之间的关系
        self.embedding = keras.layers.Embedding(cfg.input_vocab_size, cfg.d_model)
        # 位置编码
        self.pos_embedding = Utils.positional_encoding(cfg.max_seq_len, cfg.d_model)  # [1, max_seq_len, d_model]
        # encoder层，可以有多个encoderLayer
        self.encode_layer = [EncoderLayer(cfg) for _ in range(cfg.n_layers)]  # n_layers = 2
        # dropout层
        self.dropout = keras.layers.Dropout(cfg.dropout_rate)

    def call(self, inputs, training=True):
        """
        :param inputs: [-1, seq_len]
        :param training: 是否训练， True: 是， False: 否
        :return:
        """
        # src mask
        mask = tf.equal(inputs, 0)  # [-1, seq_len]
        mask = mask[:, :, np.newaxis]  # [-1, seq_len, 1]

        word_emb = self.embedding(inputs)  # [-1, seq_len, d_model]

        # 乘以sqrt(d_model),以减少后面position encoding的影响，position encoding的最大值是1，最小值是0；
        # todo 为什么是乘以sqrt(d_model)，而不是乘以其它一个常数值呢？
        word_emb *= tf.sqrt(tf.cast(self.cfg.d_model, tf.float32))

        # 加上位置编码
        emb = word_emb + self.pos_embedding
        emb = tf.where(mask, 0, emb)

        mask = tf.cast(mask, tf.float32)
        mask = mask[:, np.newaxis, :, :]  # [-1, 1, seq_len, 1]
        mask = tf.maximum(mask, tf.transpose(mask, [0, 1, 3, 2]))
        x = self.dropout(emb, training=training)  # 添加
        for i in range(self.cfg.n_layers):
            x = self.encode_layer[i](x, mask, training)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, cfg: Config):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(cfg.d_model, cfg.n_heads)
        self.ffn = feed_forward_network(cfg.d_model, cfg.ddf)

        self.norm_1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_1 = tf.keras.layers.Dropout(cfg.dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(cfg.dropout_rate)

    def call(self, inputs, mask, training):
        """
        :param inputs: [-1, seq_len, d_model]
        :param mask: [-1, 1, seq_len, seq_len]
        :return:
        """
        # 多头注意力网络
        att_output = self.mha(inputs, inputs, inputs, mask)
        att_output = self.dropout_1(att_output, training=training)
        out1 = self.norm_1(inputs + att_output)  # (batch_size, input_seq_len, d_model)
        # 前向网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout_2(ffn_output, training=training)
        out2 = self.norm_2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2
