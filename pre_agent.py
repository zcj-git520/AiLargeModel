from typing import Dict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_store
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt, Command
from langgraph_supervisor import create_supervisor

from common.modeCommon import Model
from config.config import Config
from tool import ip_to_city_tool, search_weather_tool, search_city_tool, get_city_by_ip

conf = Config("conf/config.yml")
model = Model(conf)
# 模型 init_chat_model
# init_chat_model(
#
# )
##### 预构建agent
"""
agent是由：三个组件组成：一个大型语言模型 (LLM)、一套可供其使用的工具(tools)，以及一个提供指令的提示(prompt)。
1、大型语言模型 (LLM)：这是一个强大的语言模型，能够理解和生成文本。LLM 在一个循环中运行。在每次迭代中，它会选择一个要调用的工具，提供输入，
接收结果（一个观察），并利用该观察来指导下一个动作。循环会一直持续，直到满足停止条件——通常是代理收集到足够的信息来响应用户时
2、工具：这是一个或多个函数或 API，用于执行特定任务，例如搜索互联网、查询数据库或调用其他服务。
3、提示：这是一个文本字符串，用于指示模型执行特定任务。
"""
"""
LangGraph 包含构建健壮、可投入生产的代理系统所需的几项基本功能：
1、内存集成：原生支持短期（基于会话）和长期（跨会话持久）内存，实现在聊天机器人和助手中的有状态行为。
2、人在回路控制：执行可以无限期暂停以等待人工反馈——这与仅限于实时交互的基于 WebSocket 的解决方案不同。这允许在工作流的任何点进行异步审批、纠正或干预。
3、流式传输支持：实时流式传输代理状态、模型令牌、工具输出或组合流。
4、部署工具：包含无需基础设施的部署工具。LangGraph 平台支持测试、调试和部署。
"""
prompt = PromptTemplate(
    input_variables=[],
    template="你是一个智能小助手"
)
tools = [
    ip_to_city_tool, search_weather_tool, search_city_tool
]
agent = create_react_agent(
    model=model.qwen_llm(), tools=tools, prompt=prompt,parallel_tool_calls=False
)
### 运行agent
"""
代理可以在两种主要模式下执行
同步使用 .invoke() 或 .stream()
result = agent.invoke({"messages": [HumanMessage(content="你好")]})
print(result)
异步使用 await .ainvoke() 或 async for 与 .astream()
async for event in agent.astream({"messages": [HumanMessage(content="你好")]}):
print(event)
"""
### 输入与输出
"""
输入：
格式	       示例
字符串	{"messages": "Hello"} — 解释为 HumanMessage
消息字典	{"messages": {"role": "user", "content": "Hello"}}
消息列表	{"messages": [{"role": "user", "content": "Hello"}]}
带自定义状态	{"messages": [{"role": "user", "content": "Hello"}], "user_name": "Alice"} — 如果使用自定义 state_schema
输出：
代理输出是一个字典，包含以下键：
1、messages：执行期间交换的所有消息列表（用户输入、助手回复、工具调用）。
2、可选地，如果配置了结构化输出，则包含 structured_response。
3、如果使用自定义 state_schema，输出中还可能包含与您定义的字段对应的额外键。这些键可以保存工具执行或提示逻辑更新后的状态值。

流式输出:
代理支持流式响应，以实现更具响应性的应用程序。这包括：
1、每一步之后的进度更新
2、生成时的LLM 令牌
3、执行期间的自定义工具消息
举例：
async for event in agent.astream({"messages": [HumanMessage(content="你好")]}):
    print(event)
    print("="*20)
"""

# 最大迭代次数
"""
要控制agent执行并避免无限循环,可以设置递归限制。这定义了代理在引发 GraphRecursionError 之前可以执行的最大步骤数。
可以在运行时或通过 .with_config() 定义代理时配置 recursion_limit
举例：
运行时配置
max_iterations = 3
recursion_limit = 2 * max_iterations + 1
agent = create_react_agent(
    llm=model.qwen_llm(), tools=tools(), prompt=prompt
    )
agent.invoke({"messages": [HumanMessage(content="Hello")]},{recursion_limit: recursion_limit})
使用.with_config()配置
agent = create_react_agent(
    llm=model.qwen_llm(), tools=tools(), prompt=prompt
    ).with_config(recursion_limit=recursion_limit)
"""
###流式传输
"""
流式传输是构建响应式应用程序的关键。您可能需要流式传输以下几种类型的数据：
1、agent进度 — 在agent图中的每个节点执行后获取更新。
2、LLM 令牌 — 在语言模型生成令牌时进行流式传输。
3、自定义更新 — 在执行过程中从工具发出自定义数据（例如，“已获取 10/100 条记录”）
"""
### agent进度
"""
agent进度是指agent在执行过程中，每个节点的执行状态。要流式传输agent进度，请使用带有 stream_mode="updates" 的 stream() 或 astream() 方法。
这会在每个代理步骤后发出一个事件
举例：
同步：
for event in agent.stream({"messages": [HumanMessage(content="你好")]}, stream_mode="updates"):
    print(event)
异步：
async for event in agent.astream({"messages": [HumanMessage(content="你好")]}, stream_mode="updates"):
    print(event)
"""
for event in agent.stream({"messages": [HumanMessage(content="获取这个ip：115.239.210.27的城市信息")]}, stream_mode="updates"):
    print(event)
    print("\n")

#LLM 令牌
"""
要流式传输 LLM 生成的令牌，stream_mode="messages"
for token, metadata in agent.stream({"messages": [HumanMessage(content="获取这个ip：115.239.210.27的城市信息")]}, stream_mode="messages"):
    print("Token", token)
    print("Metadata", metadata)
"""
# 工具
"""
工具是一种封装函数及其输入模式的方式，可以将其传递给支持工具调用的聊天模型。这允许模型请求以特定输入执行此函数
@tool
def search_weather_tool(id: str) -> str:
    '''通过城市id获取天气信息'''
    url = f"http://t.weather.itboy.net/api/weather/city/{id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise ToolException(f"错误：{id}不存在，获取城市天气数据数据失败")
"""
# 禁用并行工具调用
"""
要禁用并行工具调用，设置 parallel_tool_calls=False
agent = create_react_agent(
    llm=model.qwen_llm(), tools=tools(), prompt=prompt, parallel_tool_calls=False

)
"""
# 直接返回工具结果
"""
使用 return_direct=True 可立即返回工具结果并停止agent循环
@tool(return_direct=True)
def search_weather_tool(id: str) -> str:
    '''通过城市id获取天气信息'''
    url = f"http://t.weather.itboy.net/api/weather/city/{id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise ToolException(f"错误：{id}不存在，获取城市天气数据数据失败")
"""
#强制使用工具
"""
要强制agent使用工具，设置 force_tool=True
在没有停止条件的情况下强制使用工具可能会创建无限循环。请使用以下防护措施之一：
1、将工具标记为 return_direct=True，以在执行后结束循环。
2、设置 recursion_limit 以限制执行步骤的数量。
agent = create_react_agent(
    llm=model.qwen_llm(), tools=tools(), prompt=prompt, force_tool=True
)
"""
#MCP 集成
"""
模型上下文协议 (MCP) 是一个开放协议，用于标准化应用程序如何向语言模型提供工具和上下文。
LangGraph 代理可以通过 langchain-mcp-adapters 库使用 MCP 服务器上定义的工具
"""
#使用 MCP 工具
"""
要使用 MCP 工具，需要安装 langchain-mcp-adapters 库
pip install langchain-mcp-adapters
langchain-mcp-adapters 包使代理能够使用一个或多个 MCP 服务器上定义的工具
 client = MultiServerMCPClient(
        {
            # 使用WMIC获取主板序列号
            "baseboard_serial": {
                "command": "wmic",
                 "args": ["baseboard", "get", "serialnumber"],
                "transport": "stdio"
            },
            # 使用WMIC获取CPU序列号
            "cpu_serial": {
                "command": "wmic",
                 "args": ["cpu", "get", "processorid"],
                "transport": "stdio"
            },
            # 根据ip获取城市信息
            "ip_to_city": {
                "url": "https://whois.pconline.com.cn/ipJson.jsp?json=true", 
                "transport": "streamable_http",
            },
            # 根据城市名称获取天气信息
            "city_to_weather": {
                "url": "http://t.weather.itboy.net/api/weather/city/101210101",
                "transport": "streamable_http",
            }
        }
    )
"""

# 自定义 MCP 服务器
"""
要创建您自己的 MCP 服务器，您可以使用 mcp 库。该库提供了一种定义工具并将其作为服务器运行的简单方法
mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str):
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
"""

#上下文
"""
代理通常需要除消息列表之外的更多信息才能有效运行。

上下文包括消息列表之外的任何数据，这些数据可以影响代理行为或工具执行。这可以是：
1、运行时传入的信息，如 `user_id` 或 API 凭据。
2、多步推理过程中更新的内部状态。
3、来自先前交互的持久记忆或事实
LangGraph 提供了三种提供上下文的主要方式：

类型	             描述	             可变？	生命周期
配置	           在运行开始时传入的数据	❌	    每次运行
状态	           执行期间可更改的动态数据	✅	    每次运行或对话
长期记忆 (存储)	可在对话之间共享的数据	✅	     跨对话
"""
# 配置 (静态上下文)
"""
配置是在运行时传入的静态数据，用于初始化代理或工具。
它通常包括模型参数、API 密钥、工具参数等。
配置在运行时是不可变的，不会在对话过程中改变。
"""
data = agent.invoke(
    {"messages":  [HumanMessage(content="获取这个城市信息")]},
    config={"configurable": {"ip": "115.239.210.27"}}
)
print("配置 (静态上下文)", data)
# 状态 (动态上下文)
"""
状态是在运行时更新的动态数据，用于跟踪代理的执行状态或维护内部状态。
它可以包括当前处理的消息、模型推理结果、工具调用信息等。
状态在运行时是可变的，会在对话过程中更新。
class CustomState(AgentState):
    user_name: str

agent = create_react_agent(
    # Other agent parameters...
    state_schema=CustomState,
)
"""
# 长期记忆 (存储)
"""
长期记忆 (存储) 是一种机制，用于在对话之间共享和持久化数据。
它可以用于存储用户偏好、历史交互、长期知识等。
长期记忆 (存储) 可以是内存、数据库、文件系统等。
"""
# 内存
"""
LangGraph支持两种对于构建对话代理至关重要的内存类型：
短期内存：通过在会话中维护消息历史来跟踪正在进行的对话。
长期内存：在不同会话之间存储用户特定或应用程序级别的数据。
"""
# 短期内存
"""
短期内存是指在单个会话中维护的内存，用于跟踪正在进行的对话。
它通常包括消息历史、当前状态、临时数据等。
1、在创建代理时提供checkpointer。checkpointer可以实现代理状态的持久性。
2、在运行代理时在配置中提供thread_id。thread_id是对话会话的唯一标识符。
"""
checkpointer = InMemorySaver()
agent = create_react_agent(
    model=model.qwen_llm(), tools=tools, prompt=prompt, checkpointer=checkpointer
)
# 长期内存
"""
长期内存是指在不同会话之间维护的内存，用于存储用户特定或应用程序级别的数据。
它可以是数据库、文件系统、缓存等。
1、在创建agent时配置一个存储以在调用之间持久化数据。
2、使用get_store函数从工具或提示中访问存储。
store = InMemoryStore()

store.put(
    ("users",),
    "user_123",
    {
        "name": "John Smith",
        "language": "English",
    }
)

def get_user_info(config: RunnableConfig) -> str:
    #Look up user info.
    # Same as that provided to `create_react_agent`
    store = get_store()
    user_id = config["configurable"].get("user_id")
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_user_info],
    store=store
)
"""
# 人机协作
"""
要在agent中审查、编辑和批准工具调用，您可以使用 LangGraph 内置的人机协作 (HIL) 功能，特别是 interrupt() 原语。
LangGraph 允许您无限期地暂停执行——几分钟、几小时甚至几天——直到收到人工输入。
这之所以可能，是因为代理状态已检查点到数据库中，这使得系统能够持久化执行上下文并在以后恢复工作流程，从中断处继续
"""
# 审查工具调用
"""
向工具添加人工审批步骤:
1、在工具中使用 interrupt() 暂停执行。
2、使用 Command(resume=...) 根据人工输入继续。
def book_hotel(hotel_name: str):
    #Book a hotel
    response = interrupt(
        f"Trying to call `book_hotel` with args {{'hotel_name': {hotel_name}}}. "
        "Please approve or suggest edits."
    )
    if response["type"] == "accept":
        pass
    elif response["type"] == "edit":
        hotel_name = response["args"]["hotel_name"]
    else:
        raise ValueError(f"Unknown response type: {response['type']}")
    return f"Successfully booked a stay at {hotel_name}."

checkpointer = InMemorySaver()

agent = create_react_agent(
    model=model.qwen_llm(),
    tools=[book_hotel],
    checkpointer=checkpointer,
)
config = {
   "configurable": {
      "thread_id": "1"
   }
}

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "book a stay at McKittrick hotel"}]},
    config):
    print(chunk)
    print("========")
    
for chunk in agent.stream(
    Command(resume={"type": "accept"}),
    # Command(resume={"type": "edit", "args": {"hotel_name": "McKittrick Hotel"}}),
    config
):
    print(chunk)
    print("\n")
"""
#多智能体
"""
单个智能体可能难以应对需要专门处理多个领域或管理多种工具的情况。为了解决这个问题，您可以将智能体分解为更小、独立的智能体，并将它们组合成一个多智能体系统。

在多智能体系统中，智能体之间需要进行通信。它们通过移交来实现这一点——这是一种描述将控制权移交给哪个智能体以及发送给该智能体的数据负载的原始操作。

两种最受欢迎的多智能体架构是：
1、主管——单个智能体由一个中央主管智能体协调。主管控制所有通信流和任务委派，根据当前上下文和任务要求决定调用哪个智能体。
2、群组——智能体根据其专业性动态地相互移交控制权。系统会记住哪个智能体上次处于活动状态，确保在后续交互中，对话会与该智能体恢复。
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
"""
#移交
"""
多智能体交互中一个常见的模式是**移交（handoffs）**，即一个智能体将控制权*移交*给另一个智能体。移交允许您指定：

目的地：要导航到的目标智能体
负载：要传递给该智能体的信息
这在 langgraph-supervisor（主管移交给单个智能体）和 langgraph-swarm（单个智能体可以移交给其他智能体）中都有使用。
"""














