import pandas as pd
from ali import *
from glob import glob
from tqdm import tqdm

# ali setup
from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
import torch
model_id = 'qwen/Qwen-VL-Chat'
revision = 'v1.1.0'

# 配置到指定的下载好的模型目录
model_dir = "Qwen-VL-Chat"
torch.manual_seed(1234)

# 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, bf16=True).eval()
# 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, fp16=True).eval()
# 使用CPU进行推理，需要约32GB内存
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cpu", trust_remote_code=True).eval()
# 默认使用自动模式，根据设备自动选择精度
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()

# 可指定不同的生成长度、top_p等相关超参
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)



# ali_caption
def ali_local_caption(file_path="../img_captioning/png/example.png", prompt="Give a brief caption of the picture."):
    # 第一轮对话 1st dialogue turn
    query = tokenizer.from_list_format([
        {'image': file_path},
        {'text': prompt},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)

    # print(response)
    # 图中是一名年轻女子在沙滩上和她的狗玩耍，狗的品种是拉布拉多。她们坐在沙滩上，狗的前腿抬起来，与人互动。
    return response

# 标注代码
images = sorted(glob('../img_captioning_CNNRNN/images/train/*.jpg'))
captions = []
i=0

bar = tqdm(images)
for image in bar:
    i+=1
    if i % 100 == 0:
        df = pd.DataFrame({'image': images[:i], 'caption': captions})
        df.to_csv(f'../captions_{i}.csv', index=False)
    caption = ali_local_caption(image)
    captions.append(caption)
    bar.set_description(f'Captioning {image}')

df = pd.DataFrame({'image': images, 'caption': captions})
df.to_csv('../captions.csv', index=False)
