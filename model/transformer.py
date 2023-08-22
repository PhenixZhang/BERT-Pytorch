import torch.nn as nn

from model.attention import MultiHeadedAttention
from model.utils import SublayerConnection, PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    双向Encoder = Transformer (Self-attention)
    Transformer = MultiHead_Attention + Feed_Forward + sublayer_connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        hidden: transformer中隐藏层的大小
        attn_heads: 多头注意力机制中的头数量
        feed_forward_hidden: 前馈神经网络中隐藏层大小，通常是hidden_size * 4
        dropout: dropout率
        """

        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
    