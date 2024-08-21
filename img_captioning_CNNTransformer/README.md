# Image Captioning using Transformer


## 1. 介绍

本项目旨在实现基于CNN+transformer的image-caption任务，所有代码基于[zarzouram/image_caption
ing_with_transformers](https://github.com/zarzouram/image_captioning_with_transformers/tree/main?tab=readme-ov-file#32-framework)
项目，详细介绍和环境配置请见原项目。

## 2. 运行

### 2.1. **数据集**

训练集包含2000张图片，验证集包含347张图片，caption数据由大模型生成保存于csv格式文件，共两列，具体格式如下

| image         | caption  |
|---------------|---------|
| ./images/train/0.jpg | a car's side mirror is reflecting the sunset |
| ./images/train/1.jpg    | two plates with food on them and a fork   |
| ./images/train/10.jpg    | a woman in a hat and white shirt standing in the ocean |


### 2.2. 代码修改

原始代码是在COCO数据集上训练，为适应csv格式读取以及使代码运行起来，对部分文件作出修改:

1. `code/models/IC_encoder_decoder/pe.py`: 位置编码发生维度错误，在`forward`中进行了修改。
2. `code/utils/build_vocab.py`: 新建文件，从同项目文件夹`img_captioning_CNNRNN`导入`vocab.pkl`读取 
时报错无法找到Vocabulary类，于是再导入此文件。
3. `code/utils/run_train.py`: 引入`CSVDataset`类型，新增`collate_fn`和`get_loader`函数，
修改了`main`函数中`load vocab` `dataloader` `load pretrained embeddings` 部分。
4. `code/utils/trainer.py`: 训练中首先会根据当前epoch判断是否微调`embeddings layer`，此时会
将参数重复添加到优化器导致报错，343行新增了判断。 `cptns_all`和`lens`会因数据维度不对导致报错，在367和368
增加维度操作。
