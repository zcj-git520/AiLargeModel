import cv2
import io
import base64

import httpx
import numpy as np
from PIL import Image


def file_to_base64_by_local(path):
    """
        将本地文件转为 Base64流
    :param path: 文件路径
    :return:
    """
    with open(path, "rb") as file:
        base64_data = base64.b64encode(file.read())  # base64编码
    return base64_data.decode("utf-8")

def image_to_base64_by_url(image_url):
    """
        将url图片转为 Base64流
    :param image_url: 图片url
    :return:
    """
    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
    return image_data


def base64_to_image_cv(base64_data):
    """
        将 Base64流 转为 Opencv格式图片
    :param base64_data: base64编码
    :return: Opencv格式图片
    """
    img_b64decode = base64.b64decode(base64_data)  # base64解码
    img_array = np.frombuffer(img_b64decode, np.uint8)  # 转换np序列
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 转换Opencv格式
    return img


def base64_to_image_PIL(base64_data):
    """
        将 Base64流 转为 PIL.Image格式图片
    :param base64_data: base64编码
    :return: PIL.Image格式图片
    """
    img_b64decode = base64.b64decode(base64_data)  # base64解码
    img = io.BytesIO(img_b64decode)
    img = Image.open(img)
    return img


if __name__ == '__main__':
    img_path = "C:\\Users\\Administrator\\Desktop\\R-C.jpg"
    base64code = file_to_base64_by_local(img_path)
    # print(base64code)

    url = "https://vcg03.cfp.cn/creative/vcg/800/new/VCG211167005684.jpg"
    base64_code = image_to_base64_by_url(url)
    print(base64_code)

    save_path = "metric_save.jpg"
    # 保存 OpenCV格式
    image = base64_to_image_cv(base64code)
    cv2.imwrite(save_path, image)

    # 保存PIL格式
    # image = base64_to_image_PIL(base64code)
    # image.save(save_path)
