from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph_swarm import create_swarm

from common.modeCommon import Model
from config.config import Config
from tool import search_city_tool, search_weather_tool, get_city_by_ip

# 通过主管创建多智能体
def create_agents_by_supervisor(model):
    search_city_tool_agent = create_react_agent(
        model=model.qwen_llm(), tools=[search_city_tool], prompt="你是一个城市搜索智能体",name="search_city_tool_agent"
    )
    get_weather_tool_agent = create_react_agent(
        model=model.qwen_llm(), tools=[search_weather_tool], prompt="你是一个天气搜索智能体",name="get_weather_tool_agent"
    )
    get_city_by_ip_agent = create_react_agent(
        model=model.qwen_llm(), tools=[get_city_by_ip], prompt="你是一个根据ip获取城市信息智能体",name="get_city_by_ip_agent"
    )
    supervisor = create_supervisor(
        model=model.qwen_llm(),
        agents=[search_city_tool_agent, get_weather_tool_agent, get_city_by_ip_agent],
        prompt="你是一个智能体协调器，你需要根据用户的问题，调用不同的智能体来完成任务",
    ).compile()
    # 运行
    for chunk in supervisor.stream(
        {"messages": [{"role": "user", "content": "获取ip为：115.239.210.27的城市信息和天气"}]},
        config={
            "configurable": {
                "thread_id": "1",
            }
        },
    ):
        print(chunk)
        print("========")

#群组智能体
def create_agents_by_group(model):
    search_city_tool_agent = create_react_agent(
        model=model.qwen_llm(), tools=[search_city_tool,search_weather_tool,get_city_by_ip], prompt="你是一个城市搜索智能体", name="search_city_tool_agent"
    )
    get_weather_tool_agent = create_react_agent(
        model=model.qwen_llm(), tools=[search_city_tool,search_weather_tool,get_city_by_ip], prompt="你是一个天气搜索智能体",
        name="get_weather_tool_agent"
    )
    get_city_by_ip_agent = create_react_agent(
        model=model.qwen_llm(), tools=[search_city_tool,search_weather_tool,get_city_by_ip], prompt="你是一个根据ip获取城市信息智能体",
        name="get_city_by_ip_agent"
    )
    swarm = create_swarm(
        agents=[search_city_tool_agent, get_weather_tool_agent, get_city_by_ip_agent],
        default_active_agent="search_city_tool_agent"
    ).compile()

    for chunk in swarm.stream(
            {"messages": [{"role": "user", "content": "获取ip为：115.239.210.27的城市信息和天气"}]},
    ):
        print(chunk)
        print("\n")

if __name__ == '__main__':
    config = Config('conf/config.yml')
    model = Model(config)
    create_agents_by_group(model)
