# 阿里通义千问 本地Setup使用方法
1. 配置pytorch环境：
   - python 3.8及以上版本
   - pytorch 1.12及以上版本，推荐2.0及以上版本
   - 建议使用CUDA 11.4及以上
   - 建议使用RTX3090等24G显存及以上显卡
2. 安装依赖：
    ```bash
    pip install modelscope -U
    pip install transformers accelerate tiktoken -U
    pip install einops transformers_stream_generator -U
    pip install "pillow==9.*" -U
    pip install torchvision
    pip install matplotlib -U
    ```
3. 下载模型：
    ```bash
    modelscope download --model qwen/Qwen-VL-Chat --cache_dir './cache_dir'
    ```
4. 配置模型路径，在`ali_local_caption.py`中将`model_dir`设置为`./cache_dir/qwen/Qwen-VL-Chat`（按实际下载的目录更改）。
5. 运行`ali_local_caption.py`。

# 阿里通义千问 API使用方法
1. 在[申请网址](https://help.aliyun.com/zh/dashscope/developer-reference/acquisition-and-configuration-of-api-key?spm=a2c4g.11186623.0.0.7b5728c46Soa44)中申请API Key。
2. 在`ali.py`中填入API Key。
3. 运行`ali.py`，输入图片路径与prompt，即可得到图片的标签。

# 百度Fuyu-8B API使用方法
1. 在[申请网址](https://console.bce.baidu.com/iam/#/iam/accesslist)中申请AK和SK。
2. 在`baidu.py`中填入AK和SK。
3. 运行`baidu.py`，输入图片路径，即可得到图片的标签。