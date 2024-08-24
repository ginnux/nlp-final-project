# Image Captioning using CNN-RNN

## 1. 介绍
使用原始CNN-RNN进行图像标注任务，其中图像编码器部分使用了预训练的ResNet101模型，图像解码器部分使用了LSTM网络架构

## 2. 运行

### 2.1. **数据集**

将Train.zip和Val.zip中的图片分别解压到"data/Train/"和"data/Val/"目录下，而data目录下的"train_data.json"和"val_data.json"分别对应大模型生成的训练集和验证集中图片的caption，并已转换成可供模型读取的形式

### 2.2. 模型训练

打开test.ipynb，顺次执行代码即可完成训练，输出模型保存到output目录下的encoder.pth和decoder.pth

### 2.3. 模型测试

在完成了模型训练或将encoder.pth和decoder.pth置于output目录下后，打开train.ipynb，顺次执行代码即可完成测试，对验证集中图片生成的caption输出在"output/result.json"中

### 2.4. 指标生成
还在做

## 3. 模型文件
https://pan.baidu.com/s/1hFKHTIUA2IhN1tiJt1vYfg?pwd=521n
