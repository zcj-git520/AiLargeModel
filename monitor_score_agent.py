import json

import requests
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool, ToolException

from langgraph.prebuilt import create_react_agent

from common.modeCommon import Model
from config.config import Config

# 监控评分agent
"""
获取监控指标数据的工具
"""
@tool()
def monitor_data():
    """
    监控指标数据
    """
    url = "http://192.168.41.151:9307/monitor/largeScreen"
    response = requests.get(url)
    print(response)
    if response.status_code == 200:
        data = response.json()

        # 递归移除所有层级的score字段
        def remove_score_fields(obj):
            if isinstance(obj, dict):
                # 移除当前层级的score字段
                if 'score' in obj:
                    del obj['score']
                # 递归处理子对象
                for key in list(obj.keys()):
                    remove_score_fields(obj[key])
            elif isinstance(obj, list):
                # 递归处理列表中的每个元素
                for item in obj:
                    remove_score_fields(item)

        # 执行递归移除
        remove_score_fields(data)

        return data
    else:
        raise ToolException(f"错误：获取监控数据失败")

def monitor_score_agent(model):
    """
    监控指标agent
    """
    instructions = f"""
    你是一个监控指标agent，你需要根据监控指标数据来评估系统的运行状态，并给出一个评分（满分100分），包括但不限于主机、
    数据库、容器、服务等方面的指标。
    评分标准：
    1. 主机指标 占比0.3。
    2. 数据库指标 占比0.1。
    3. 容器指标 占比0.3。
    4. 服务指标 占比0.3。
    状态信息：
    0：正常
    1：异常
    2：警告
    输出 json 格式：
    '
        "score": 0,
        "status":"运行状态【差，良好，优秀】",
        "comment": "",
        "suggest": ""
    '
    """


    tools = [monitor_data]
    input_message = HumanMessage(
        content= instructions,
    )
    stream_config = {"configurable": {"thread_id": "abc123"}}
    agent_executor = create_react_agent(model, tools)
    try:
        for step in agent_executor.stream({"messages": [input_message]}, stream_config, stream_mode="values"):
            step["messages"][-1].pretty_print()
    except Exception as e:
        print(f"智能体执行出错: {str(e)}")


if __name__ == '__main__':
    config = Config("conf/config.yml")
    model = Model(config)
    monitor_score_agent(model.qwen_llm())

