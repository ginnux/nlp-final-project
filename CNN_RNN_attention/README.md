# CNN-RNN with Attention

## 1. 介绍
使用带有soft attention的CNN-RNN进行图像标注任务，其中图像编码器部分使用了预训练的ResNet152模型，序列解码器部分使用了LSTM网络架构

## 2. 运行

### 2.1. **数据集**

将Train.zip和Val.zip中的图片分别解压到"data/Train/"和"data/Val/"目录下，再运行`build_vocab.py`。

### 2.2. 模型训练

运行`train.py`，得到loss等可视化曲线和保存的模型权重文件。

### 2.3. 模型测试与指标生成

运行`evaluate.py`, 输出结果的BLEU,GLEU,Meteor指标并得到val数据集上的输出到`output.csv`（需要安装nltk以及相关的支持库）
