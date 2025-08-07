import aiohttp
from langchain_core.tools import ToolException
from mcp.server import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str):
    """
    异步获取指定城市的天气信息。

    :param location: 城市的天气代码，例如 "101020100"
    :return: 包含天气信息的 JSON 数据
    :raises ToolException: 如果请求失败，抛出异常
    """
    url = f"http://t.weather.itboy.net/api/weather/city/{location}"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise ToolException(f"错误：{location} 不存在，获取城市天气数据失败，状态码: {response.status}")
        except Exception as e:
            raise ToolException(f"错误：获取 {location} 天气数据时发生网络错误: {str(e)}")

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
