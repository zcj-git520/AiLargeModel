# 自定义工具调用
"""
工具的几个要素：
工具名称
工具描述
该工具接收的 JSON 格式规范
需调用的函数
工具结果是否直接反馈给用户
"""

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import GoogleCloudTextToSpeechTool

def get_wikipedia(query):
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
    tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    result = tool.run(query)
    return result

if __name__ == '__main__':

    # 可以用字典输入来调用这个工具
    print(get_wikipedia({"query": "langchain"}))
    # 使用单个字符串输入来调用该工具。
    print(get_wikipedia("langchain"))
    text_to_speak = "Hello world!"

    tts = GoogleCloudTextToSpeechTool()
    speech_file = tts.run(text_to_speak)
