import tensorflow as tf
import numpy as np
import math


def positional_encoding(position, d_model):
    """"
    位置编码：因为CNN没有时间序列，添中位置信息，让特征包含位置信息
    :param position: 位置大小
    :param d_model: 维度大小
    :return: tensor: [1, position, d_model]
    """""
    # 位置编码
    pe = tf.Variable(tf.zeros(shape=[position, d_model]))  # [max_len, d_model]
    # 位置
    position = tf.reshape(tf.range(0, position, 1, dtype=tf.float32), shape=[-1, 1])  # [max_len, 1]
    # 计算 1 / (1000^(2i / d))
    # 涉及的运算公式：x = e^ln(x), ln(x^a) = a * ln(x)
    div_term = tf.exp(
        tf.range(0, d_model, 2, dtype=tf.float32) * (-math.log(10000.0) / d_model))  # [d_model/2,]
    # 双数位置使用sin函数， 单数位置使用cos函数
    pe[:, 0::2].assign(tf.sin(position * div_term))
    pe[:, 1::2].assign(tf.cos(position * div_term))
    # 因为x的维度是[-1, position, d_model]，所以位置编码的维度也要跟x一样，才能做广播相加
    pos_encoding = pe[np.newaxis, ...]  # [1, position, d_model]

    return tf.cast(pos_encoding, dtype=tf.float32)  # 转换类型后返回


def create_look_ahead_mark(seq_len):
    """"
    decoder mask multi head attention 的 mask
    对角线和下三角全部为0，其它为1
    :param seq_len:
    :return: eg: [0  1  1
                  0  0  1
                  0  0  0]
    """""
    mark = 1 - tf.linalg.band_part(input=tf.ones((seq_len, seq_len)),
                                   num_lower=-1,  # 负数，下三角全部保留
                                   num_upper=0)  # 上三角保留0个
    return mark  # (seq_len, seq_len)


def decoder_mask(inputs):
    """"
    used decoder mask multi head attention
    :param inputs: [-1, max_len]
    :return: [-1, max_len, max_len]
    """""
    # look_ahead 掩码，掩掉未预测的词
    seq_len = tf.shape(inputs)[1]
    look_ahead_mask = create_look_ahead_mark(seq_len)  # [max_len, max_len]
    # decoder第一层得到padding掩码
    padding_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    padding_mask = padding_mask[:, :, np.newaxis]
    # 合并decoder第一层掩码，用于decoder中的mask-multi-head-attention
    return tf.maximum(padding_mask, look_ahead_mask)  # [-1, max_len, max_len]
