import os
import qianfan
import base64
from qianfan.resources import Image2Text

# 使用安全认证AK/SK鉴权，通过环境变量方式初始化；替换下列示例中参数，安全认证Access Key替换your_iam_ak，Secret Key替换your_iam_sk
# 在申请网址中获取AK和SK，申请网址：https://console.bce.baidu.com/iam/#/iam/accesslist
os.environ["QIANFAN_ACCESS_KEY"] = "your_iam_ak"
os.environ["QIANFAN_SECRET_KEY"] = "your_iam_sk"

def baidu_caption(file_path="../img_captioning/png/example.png", prompt="Give a brief caption of the picture."):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    # 使用model参数
    i2t = Image2Text(model="Fuyu-8B")
    resp = i2t.do(prompt=prompt, image=encoded_string)

    return resp["result"]