import asyncio
import aiohttp
import requests
from langchain_core.tools import ToolException

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from mcp.server import FastMCP

from common.modeCommon import Model
from config.config import Config
from tool import search_weather_tool


async def async_get_tools(model):
    # 初始化MCP客户端
    client = MultiServerMCPClient(
        {
            # # 使用WMIC获取主板序列号
            # "baseboard_serial": {
            #     "command": "wmic",
            #      "args": ["baseboard", "get", "serialnumber"],
            #     "transport": "stdio"
            # },
            # # 使用WMIC获取CPU序列号
            # "cpu_serial": {
            #     "command": "wmic",
            #      "args": ["cpu", "get", "processorid"],
            #     "transport": "stdio"
            # },
            # # 根据ip获取城市信息
            # "ip_to_city": {
            #     "url": "https://whois.pconline.com.cn/ipJson.jsp?json=true",
            #     "transport": "streamable_http",
            # },
            # 根据城市名称获取天气信息
            "get_weather": {
                "url": "https://127.0.0.1:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )


    try:
        print("准备调用 client.get_tools() 方法...")
        # 异步获取工具
        tools = await client.get_tools()
        print(f"client.get_tools() 方法调用完成，获取到的工具列表: {tools}")
        if tools is None:
            print("警告：获取到的工具列表为 None")
            return None
        if not tools:
            print("警告：获取到的工具列表为空")
        return create_react_agent(
            model.qwen_llm(),
            tools
        )
    except Exception as e:
        print(f"获取工具时出错: {e}")
        raise

def get_mcp_agent(model):
    try:
        agent = asyncio.run(async_get_tools(model))
        if agent:
            async def run_agent():
                baseboard_serial_response = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "101020100的天气情况"}]}
                )
                print(baseboard_serial_response)
            asyncio.run(run_agent())
        else:
            print("未获取到有效的代理，无法执行查询。")
    except Exception as e:
        print(f"程序运行出错: {e}")





if __name__ == '__main__':
    conf = Config("conf/config.yml")
    model = Model(conf)
    get_mcp_agent(model)
