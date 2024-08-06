from http import HTTPStatus
import dashscope
import os

# 设置API_KEY
# 申请网址：https://help.aliyun.com/zh/dashscope/developer-reference/acquisition-and-configuration-of-api-key?spm=a2c4g.11186623.0.0.7b5728c46Soa44
api_key = "API_KEY"
dashscope.api_key = api_key

if api_key == "API_KEY":
    raise ValueError("请设置API_KEY，申请网址见：https://help.aliyun.com/zh/dashscope/developer-reference/acquisition-and-configuration-of-api-key?spm=a2c4g.11186623.0.0.7b5728c46Soa44")


def simple_multimodal_conversation_call(file_path,prompt="Give a brief caption of the picture."):
    messages = [
        {
            "role": "user",
            "content": [
                {"image": file_path},
                {"text": prompt}
            ]
        }
    ]
    response = dashscope.MultiModalConversation.call(model=dashscope.MultiModalConversation.Models.qwen_vl_chat_v1,
                                                     messages=messages,
                                                     )

    if response.status_code == HTTPStatus.OK:  #如果调用成功，则打印response
        # print(response)
        return response
    else:  #如果调用失败
        print(response.code)  # 错误码
        print(response.message)  # 错误信息
        exit(1)


def ali_caption(file_path="../img_captioning/png/example.png", prompt="Give a brief caption of the picture."):
    # 求绝对本地文件路径
    abs_path = os.path.abspath(file_path)
    local_file_path = f'file://{abs_path}'

    # 调用多模态对话接口
    response = simple_multimodal_conversation_call(local_file_path)

    return response["output"]["choices"][0]["message"]["content"]


if __name__ == '__main__':
    caption = ali_caption()
    print(caption)
