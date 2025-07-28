# 多模态输入
import base64
import json

from langchain_core.messages import HumanMessage

from common.modeCommon import Model
from config.config import Config
from utils import images_util


class Multimodal(object):
    def __init__(self, model: Model):
        self.model = model

    def input_by_base64(self, prompt_txt, image_data, type):
        try:

            message = HumanMessage(
                content=json.dumps([
                    {"type": "text", "text": prompt_txt},
                    {"type": type, "data": image_data, "source_type": "base64"},
                ])
            )

            china = self.model.qwen_llm()
            return china.invoke([message]).content
        except Exception as e:
            print(e)
            return ""

    def input_by_url(self, prompt_txt, url, type):

        try:
            message = HumanMessage(
                content=json.dumps([
                    {"type": "text", "text": prompt_txt},
                    {"type": type, "data": url, "source_type": "url"},
                ])
            )
            china = self.model.qwen_llm()
            return china.invoke([message]).content
        except Exception as e:
            print(e)
            return ""

if __name__ == '__main__':
    config = Config('conf/config.yml')
    # 初始化模型
    model = Model(config)
    multimodal = Multimodal(model)
    # image_path = "C:\\Users\\Administrator\\Desktop\\17536735231103.png"
    # base64_code = images_util.image_to_base64_by_local(image_path)

    # result = multimodal.input_by_image_base64("这张图片描述了什么,中文描述", base64_code, "image/png")
    # result = multimodal.input_by_url("这张图片描述了什么,中文描述", "https://img.shetu66.com/2023/04/25/1682391094827084.png", "image")
    file_path = "C:\\Users\\Administrator\\Desktop\\1.pdf"
    base64_file = images_util.file_to_base64_by_local(file_path)
    result = multimodal.input_by_base64("文件主要讲了什么", base64_file, "file")
    print(result)
