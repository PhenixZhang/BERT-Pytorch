# 本库使用Pytorch实现BERT源码

其中dataset中包含数据集处理以及词表生成

model中包含attention、embedding、MLM&NSP乃至transformer块的源码实现

trainer中包含优化器学习率调整以及pretrain流程代码

可直接在外层搭配示例运行即可

## Step 1：生成词表

```shell
python dataset/vocab.py -c test.txt -o test.vocab
```

## Step 2：产出模型并保存

```shell
python main.py -v test.vocab -c test.txt -o bert.model --with_cuda False
```