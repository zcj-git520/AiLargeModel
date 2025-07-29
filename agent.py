"""
LangChain 支持创建智能体，即使用大型语言模型作为推理引擎来决定采取哪些行动以及执行行动所需的输入。
执行行动后，可以将结果反馈给大型语言模型，以判断是否需要更多行动，或者是否可以结束。这通常通过工具调用实现。
"""
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from common.modeCommon import Model
from config.config import Config
from tool import ip_to_city_tool, search_city_tool, search_weather_tool


# 创建一个查询通过ip查询城市天气情况的智能体
def ip_weather_agent(model: Model, tools:list):
    memory = MemorySaver()
    llm_model = model.qwen_llm()
    agent_executor = create_react_agent(llm_model, tools, checkpointer=memory)
    return agent_executor

if __name__ == '__main__':
    config = Config('conf/config.yml')
    # 初始化模型
    model = Model(config)
    tools = [search_city_tool, search_weather_tool, ip_to_city_tool]
    agent_executor = ip_weather_agent(model, tools)
    input_message = HumanMessage(
        content="我有一个ip,ip地址为：115.239.210.27，请告诉我这个ip的城市属于贵州省的么"
    )
    stream_config = {"configurable": {"thread_id": "abc123"}}
    try:
        for step in agent_executor.stream({"messages": [input_message]}, stream_config, stream_mode="values"):
            step["messages"][-1].pretty_print()
    except Exception as e:
        print(f"智能体执行出错: {str(e)}")