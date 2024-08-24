# nlp-final-project
Natural Language Processing final project.

## 1. 介绍

### 1.1. **任务1 - 图片描述生成**

使用已有的多模态大模型为图片生成多维度描述信息。设计合适的prompt,鼓励尝试不同的prompt,比较生成
的描述有何不同,总结优秀prompt的写作技巧。

### 1.2. **基于深度学习的图片描述自动生成**

用生成的图片描述作为标注,利用 CNN-RNN 或其他常见的图像描述生成模型架构,在训练集上训练图像描述
模型,并在验证集上进行验证。加入注意力机制并通过实验讨论加入注意力机制前后模型性能的变化。模型性
能评估可以使用传统的指标如 BLEU、METEOR、CIDEr 等。

## 2. 实践

### 2.1. **大模型选用**

1. 阿里云:qwen-vl-chat（本地搭建）
2. Adept AI:Fuyu-8B（API）
3. Salesforce: Blip2、Blip（本地搭建）

### 2.2. **模型架构**

1. CNN-RNN
2. CNN-RNN with attention
3. CNN-Transformer
