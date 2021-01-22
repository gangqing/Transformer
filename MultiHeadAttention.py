import tensorflow as tf


# 构造multi head attention层
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        """
        :param d_model: 词向量维度
        :param num_heads: multi-head num
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # d_model 必须可以正确分为各个头
        assert d_model % num_heads == 0
        # 分头后的维度
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q_inputs, k_inputs, v_inputs, mask):
        """
        :param q_inputs: [batch_size, seq_len, d_model]
        :param k_inputs: [batch_size, seq_len, d_model]
        :param v_inputs: [batch_size, seq_len, d_model]
        :param mask: [batch_size, seq_len, 1]
        :return:
        """
        batch_size = tf.shape(q_inputs)[0]

        # 分头前的前向网络，获取q、k、v语义
        q = self.wq(q_inputs)  # (batch_size, seq_len, d_model)
        k = self.wk(k_inputs)
        v = self.wv(v_inputs)

        # 分头
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # 通过缩放点积注意力层
        scaled_attention = scaled_dot_product_attention(q, k, v, mask)

        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)
        # 合并多头
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # 全连接重塑
        output = self.dense(concat_attention)
        return output


def scaled_dot_product_attention(q, k, v, mask):
    """
    :param q: [-1, n_head, seq_len, d_model//n_head]
    :param k: [-1, n_head, seq_len, d_model//n_head]
    :param v: [-1, n_head, seq_len, d_model//n_head]
    :param mask: [-1, 1, seq_len, 1]
    :return:
    """
    # query key 相乘获取匹配关系
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 使用dk进行缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 掩码
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 通过softmax获取attention权重
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # todo
    mask = tf.cast(mask, tf.bool)
    attention_weights = tf.where(mask, 0, attention_weights)
    # attention 乘上value
    output = tf.matmul(attention_weights, v)  # （.., seq_len_v, depth）

    return output