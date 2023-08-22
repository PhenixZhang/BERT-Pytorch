import math

import torch
import torch.nn as nn

class SublayerConnection(nn.Module):
    """
    子层连接模块，在两个子层中每个子层应用残差连接并进行层归一化
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "将残差连接应用于每一个相同大小的子层"
        return x + self.dropout(sublayer(self.norm(x)))
    

class LayerNorm(nn.Module):
    """
    LayerNorm模块，每条样本的所有特征归一化，区别于BatchNorm中的每个batch进行归一化
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    

class PositionwiseFeedForward(nn.Module):
    "实现FFN前馈神经网络"
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU() # torch.nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    

class GELU(nn.Module):
    """
    论文3.4部分，注明BERT使用GELU激活函数替换RELU，torch当前已经支持GELU，torch.nn.GELU
    GELU函数优点：
        1)处理负数时不会像ReLU一样将输入裁剪到0，并且导数更为光滑，从而减少了训练过程中出现的梯度消失问题
        2)GELU函数在激活函数的非线性变换中引入了类似于sigmoid函数的变换，这使得GELU函数的输出可以落在一个更广的范围内，有助于加速模型的收敛速度
    """
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))