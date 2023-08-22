import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from model.bert import BERT
from model.language_model import BERTLM
from trainer.optim_schedule import ScheduledOptim

import tqdm

class BERTTrainer:
    """
    BERTTrainer通过两种语言模型训练方法构建预训练BERT模型
        1) MLM（Masked Language Model, 掩码语言模型, 3.3.1 Task #1: Masked LM）
        2) NSP(Next Sentence Prediction Model, 下一句子预测, 3.3.2 Task #2: Next Sentence Prediction)
    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps: int = 10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        bert: 你想要训练的BERT模型
        vocab_size: 全部词表大小
        train_dataloader: 训练数据集的dataloader
        test_dataloader: 测试数据集的dataloader，可以置空(None)
        lr: 优化器的学习率
        betas: Adam优化器的超参数betas
        weight_decay: Adam optimizer weight decay param Adam的权重衰减参数
        with_cuda: 使用cuda(GPU显卡)训练
        log_freq: 在batch迭代中的日志频率
        """

        # 为BERT设置cuda设备，参数为 -c， --cuda应该为true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # BERT模型会每一个epoch保存一次
        self.bert = bert
        # 使用BERT模型初始化BERT语言模型
        self.model = BERTLM(bert, vocab_size).to(self.device)

        # 如果cuda能够检测到超过一台GPU，就进行分布式GPU训练
        if with_cuda and torch.cuda.device_count() > 1:
            print("使用 %d GPUs训练BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # 设置训练集和测试集dataloader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # 设置带有超参数的Adam优化器
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # 使用负对数似然损失函数预测被mask的token
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq

        print("全部参数量：", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        循环遍历data_loader以进行训练或测试
        如果在训练状态，反向操作被激活，并每一个epoch自动保存一次模型
        epoch: 当前epoch索引
        data_loader: torch.utils.data.DataLoader子类用于构建训练/测试数据流
        train: 训练或者测试开关
        return: None
        """
        str_code = "train" if train else "test"

        #  设置tqdm进度条
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            # batch_data将被发送到设备(GPU或cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 前向计算next_sentence_prediction和masked_lm模型
            next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])

            # is_next分类结果的NLL(负对数似然)损失
            next_loss = self.criterion(next_sent_output, data["is_next"])

            # 预测mask的token对应的NLL(负对数似然)损失
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

            # next_loss和mask_loss相加
            loss = next_loss + mask_loss

            # 只在训练过程中反向传播并且优化
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # 下一句子预测准确率
            correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, 平均损失=" % (epoch, str_code), avg_loss / len(data_iter), "总损失=",
              total_correct * 100.0 / total_element)
        
    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        将当前BERT模型保存在file_path中

        epoch: current epoch number当前epoch数
        file_path: model output path which gonna be file_path+"ep%d" % epoch 模型输出路径，会被拼接epoch数
        return: final_output_path 最终输出路径
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d 模型保存在: " % epoch, output_path)
        return output_path