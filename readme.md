基于TensorFlow 2.0实现的Transformer框架。

输入的数据是统一长度的句子，假设该长度是max_len，超出该长度的句子应该截取，
小于该长度的句子，在后面补0直到句子长度为max_len。
