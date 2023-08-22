import torch.nn as nn

from model.bert import BERT

class BERTLM(nn.Module):
    """
    BERT语言模型
    MLM（Masked Language Model, 掩码语言模型）+NSP(Next Sentence Prediction Model, 下一句子预测)
    其中MLM等同于完形填空，掩码语言模型能够捕捉到文字、词汇和句法等不同层面的语言规律，并在有监督的任务中取得更好的性能
    而NSP任务可能并不是必要的，消除NSP损失在下游任务的性能上能够与原始BERT持平或略有提高。这可能是由于Bert以单句子为单位输入，模型无法学习到词之间的远程依赖关系。针对这一点，后续的RoBERTa、ALBERT、spanBERT都移去了NSP任务
    
    """
    def __init__(self, bert: BERT, vocab_size):
        """
        bert: 需要训练的BERT模型
        vocab_size: 掩码语言模型的总词表大小
        """

        super(BERTLM, self).__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_model):
        x = self.bert(x, segment_model)
        return self.next_sentence(x), self.mask_lm(x)
    
class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    二分类模型: 是下一句 or 不是下一句
    """

    def __init__(self, hidden):
        """
        hidden: BERT模型输出尺寸
        """
        super(NextSentencePrediction, self).__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))
    

class MaskedLanguageModel(nn.Module):
    """
    从被屏蔽的输入序列预测原始标记，n类分类问题，N-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        hidden: BERT模型的输出大小
        vocab_size: 总词表大小
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))