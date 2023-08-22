import math

import torch
import torch.nn as nn

class BERTEmbedding(nn.Module):
    """
    BERT Embedding层由以下特征组成：
        1) TokenEmbedding: 常规embedding矩阵
        2) PositionalEmbedding: 使用正/余弦函数添加位置信息
        3) SegmentEmbedding: 添加句子分割信息(sent_A:1, sent_B:2)
    所有特征求和就是BERTEmbedding的输出
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        vocab_size: 词汇表大小
        embed_size: 词嵌入维度
        dropout: dropout比例
        """
        super(BERTEmbedding, self).__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)
    

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):
    """
    实现PE功能，尝试过参数学习版本的PE，但是最终选择了正弦版本
    因为它可以允许模型推断出比训练期间遇到的序列长度更长的序列长度
    """
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 在对数空间中计算一次位置encoding
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # pe-tensor中0维代表batchsize，1维代表token，2维代表embedding
        x = self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

    
class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)