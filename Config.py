

class Config:
    def __init__(self):
        self.input_vocab_size = 8500  # 最大的词类种数，例如样本的有8500个不同的词
        self.target_vocab_size = 3  # 输出的维度大小
        self.d_model = 2  # 词向量大小
        self.max_seq_len = 3  # 样本的最大长度
        self.dropout_rate = 0.0  # dropout
        self.n_layers = 2  # 2层encoder_layers和decoder_layers
        self.n_heads = 1  # multi-headed attention num
        self.ddf = 2  # feed forward filters
