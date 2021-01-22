import tensorflow as tf
from tensorflow import keras
from Config import Config
from Encoder import Encoder
from Decoder import Decoder
import numpy as np


class Transformer(keras.Model):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        self.encoder = Encoder(config)  # 编码器
        self.decoder = Decoder(config)  # 解码器
        self.final_layer = tf.keras.layers.Dense(config.target_vocab_size)  # 最后输出层

    def call(self, enc_inputs, dec_inputs, targets, training=True):
        """
        :param enc_inputs: [-1, word_len]
        :param dec_inputs: [-1, targets_word_len]
        :param targets: [-1, targets_word_len]
        :param training: True : 训练; False : 测试
        :return: [-1, targets_word_len]
        """
        encode_out = self.encoder(enc_inputs, training)
        decode_out = self.decoder(enc_inputs, encode_out, dec_inputs, targets, training)
        final_out = self.final_layer(decode_out)
        return final_out


if __name__ == '__main__':
    cfg = Config()
    transformer = Transformer(cfg)

    enc_inputs = np.array([[1, 2, 3], [2, 1, 0]])
    dec_inputs = np.array([[1, 2, 3], [2, 1, 3]])
    targets = np.array([[1, 2, 3], [2, 1, 3]])

    # transformer.compile()
    # transformer.fit()

    out = transformer(enc_inputs, dec_inputs, targets)
    print(out)