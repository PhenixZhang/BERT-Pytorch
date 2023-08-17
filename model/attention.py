import copy
import math

import torch
import torch.nn as nn


def clones(module, N):
    "生成多个相同的层"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "计算归一化点乘注意力权值：Scaled Dot Product Attention（SDPA）"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    多头注意允许模型在不同位置共同注意来自不同表示子空间的信息。对于单一注意力头，平均会抑制这一点
        1）解码器中的每个位置都能获取输入序列中的所有位置
        2）编码器中的每个位置都可以处理编码器前一层中的所有位置
        3）解码器中的每个位置能获取解码器之前的所有位置直至并包括该位置（-∞）
    """
    def __init__(self, h, d_model, dropout=0.1):
        "初始化模型size和head头个数"
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # 假定d_v = d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "实现多注意力头"
        if mask is not None:
            # 对所有h个head应用相同mask策略
            mask = mask.unsqueeze(1) # 增加一维
        nbatches = query.size(0)

        # 1) 多头qkv映射，其中linears只用前三层
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        # 2) 应用attention到batch中的所有映射向量 
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) 用view进行拼接并且应用最后一个线性映射，注意contiguous写法
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query, key, value
        
        return self.linears[-1](x)