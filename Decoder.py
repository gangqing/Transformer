import tensorflow as tf
from tensorflow import keras
from Config import Config
from MultiHeadAttention import MultiHeadAttention
from FeedForward import feed_forward_network
import Utils
import numpy as np


class Decoder(keras.layers.Layer):
    def __init__(self, cfg: Config):
        super(Decoder, self).__init__()
        self.cfg = cfg

        self.embedding = keras.layers.Embedding(cfg.input_vocab_size, cfg.d_model)
        self.pos_embedding = Utils.positional_encoding(cfg.max_seq_len, cfg.d_model)  # [1, 40, 128]

        self.decoder_layers = [DecoderLayer(cfg) for _ in range(cfg.n_layers)]

        self.dropout = keras.layers.Dropout(cfg.dropout_rate)

    def call(self, enc_inputs, encode_out, dec_inputs, targets, training=True):
        """
        :param enc_inputs: [batch_size, seq_len]
        :param encode_out: [batch_size, seq_len, d_model]
        :param dec_inputs: [batch_size, seq_len]
        :param targets: [batch_size, seq_len]
        :param training: True: train; False: test
        :return:
        """
        look_ahead_mask = Utils.decoder_mask(dec_inputs)
        # src mask
        mask = tf.equal(dec_inputs, 0)  # [-1, seq_len]
        mask = mask[:, :, np.newaxis]  # [-1, seq_len, 1]

        word_emb = self.embedding(dec_inputs)
        word_emb *= tf.sqrt(tf.cast(self.cfg.d_model, tf.float32))

        # 加上位置编码
        emb = word_emb + self.pos_embedding
        emb = tf.where(mask, 0, emb)

        padding_mask = tf.cast(mask, tf.float32)
        padding_mask = padding_mask[:, np.newaxis, :, :]  # [-1, 1, seq_len, 1]
        padding_mask = tf.maximum(padding_mask, tf.transpose(padding_mask, [0, 1, 3, 2]))
        h = self.dropout(emb, training=training)
        # 叠加解码层
        for i in range(self.cfg.n_layers):
            h = self.decoder_layers[i](h, encode_out, look_ahead_mask, padding_mask, training)
        return h


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, cfg: Config):
        super(DecoderLayer, self).__init__()
        self.cfg = cfg

        self.mha_1 = MultiHeadAttention(cfg.d_model, cfg.n_heads)
        self.mha_2 = MultiHeadAttention(cfg.d_model, cfg.n_heads)

        self.ffn = feed_forward_network(cfg.d_model, cfg.ddf)

        self.norm_1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_3 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_1 = keras.layers.Dropout(cfg.dropout_rate)
        self.dropout_2 = keras.layers.Dropout(cfg.dropout_rate)
        self.dropout_3 = keras.layers.Dropout(cfg.dropout_rate)

    def call(self, inputs, encode_out, look_ahead_mask, padding_mask, training):
        """
        :param inputs:
        :param encode_out:
        :param mask: [batch_size, max_len, max_len]
        :param training:
        :return:
        """
        # masked multi-head attention
        look_ahead_mask = look_ahead_mask[:, np.newaxis, :, :]
        att1 = self.mha_1(inputs, inputs, inputs, look_ahead_mask)
        att1 = self.dropout_1(att1, training=training)
        out1 = self.norm_1(inputs + att1)
        # todo
        att2 = self.mha_2(encode_out, encode_out, inputs, padding_mask)
        att2 = self.dropout_2(att2, training=training)
        out2 = self.norm_2(out1 + att2)

        ffn_out = self.ffn(out2)
        ffn_out = self.dropout_3(ffn_out, training=training)
        out3 = self.norm_3(out2 + ffn_out)

        return out3
