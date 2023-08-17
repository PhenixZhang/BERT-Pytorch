import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import BERTEmbedding


class BERT(nn.Module):
    """
    BERT模型：基于Transformers的双向Encoder表示
    """
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        vocab_size: 全部word的词表大小
        hidden: BERT模型的隐藏层大小
        n_layers: Transformer块的数量（层数）
        attn_heads: 注意力头的个数
        dropout: dropout率
        """
        
        super(BERT, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # 论文注明在ff_network_hidden_size上使用了4*hidden_size
        self.ff_network_hidden_size = 4 * hidden

        # BERT中的embedding = pos_emb + seg_emb + tok_emb
        self.embedding = BERTEmbedding(vocab_size, hidden)

        # 多层transformer块的深层网络
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)]
        )

    def forward(self, x, segment_info):
        """
        x: 输入序列
        segment_info: 输入序列的segment_ids
        """

        # attention中对于padding的mask
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)


        # 将索引序列嵌入到向量序列
        x = self.embedding(x, segment_info)

        # 在多个transformer块上运行
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
            
        return x