# 自定义工具调用
"""
当构建自己的代理时，您需要为其提供一组工具列表，代理可以使用这些工具。除了调用的实际函数之外，工具由几个组件组成：

name（str），是必需的，并且在提供给代理的工具集中必须是唯一的
description（str），是可选的但建议的，因为代理使用它来确定工具的使用方式
return_direct（bool），默认为 False
args_schema（Pydantic BaseModel），是可选的但建议的，可以用于提供更多信息（例如，few-shot 示例）或用于验证预期参数。
"""
import requests
from charset_normalizer.api import handler
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import GoogleCloudTextToSpeechTool
from langchain_core.tools import tool, StructuredTool, ToolException

"""
内置工具包
"""


def get_wikipedia(query):
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
    tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    result = tool.run(query)
    return result


def google_tts(text):
    tts = GoogleCloudTextToSpeechTool()
    speech_file = tts.run(text)
    return speech_file


"""
自定义工具调用
"""


# 使用@tool装饰器 -- 定义自定义工具的最简单方式。
@tool
def search_city_tool(city_code: str) -> str:
    """通过城市code获取城市信息"""
    url = f"https://api.songzixian.com/api/china-city?dataSource=LOCAL_CHINA_CITY&code={city_code}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise ToolException(f"错误：{city_code}不存在，获取城市数据失败")


@tool
def search_weather_tool(id: str) -> str:
    """通过城市id获取天气信息"""
    url = f"http://t.weather.itboy.net/api/weather/city/{id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise ToolException(f"错误：{id}不存在，获取城市天气数据数据失败")
        # return ""


# StructuredTool.from_function 类方法提供了比@tool装饰器更多的可配置性，而无需太多额外的代码。

def get_city_by_ip(ip: str):
    """
    根据IP地址获取城市信息
    """
    url = f"https://whois.pconline.com.cn/ipJson.jsp?ip={ip}&json=true"
    response = requests.get(url)
    try:
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        # print(e)
        raise ToolException(f"错误：ip获取城市数据失败，{ip}")


@tool
def ip_to_city_tool(ip):
    """
    根据IP地址获取城市信息
    """
    return get_city_by_ip(ip)


def ip_to_city_tool2(ip: str):
    calculator = StructuredTool.from_function(func=get_city_by_ip, handle_tool_error="ip获取城市数据失败")
    try:
        calculator.invoke({"ip": ip})
        return calculator
    except Exception as e:
        print(e)
        return ""


if __name__ == '__main__':
    print(search_city_tool.name)
    print(search_city_tool.description)
    print(search_city_tool.args_schema)
    print(search_city_tool("杭州"))
    print(search_weather_tool("101020100"))

    print(ip_to_city_tool("115.239.210.27"))
