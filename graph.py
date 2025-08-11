from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from sqlalchemy.sql.annotation import Annotated
from typing_extensions import TypedDict

from common.modeCommon import Model
from config.config import Config
from tool import ip_to_city_tool


class State(TypedDict):
    messages: Annotated[list, add_messages]


def node(state: State):
    messages = state["messages"]
    response = model.qwen_llm().invoke(messages)
    return {"messages": [response]}

def buildGraph():
    graph_builder = StateGraph(State)
    memory = MemorySaver()
    # llm_with_tools = model.qwen_llm().bind_tools(tools())
    graph_builder.add_node("chatbot", node)
    tool_node = ToolNode(tools=tools())
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_edge(START, "chatbot")

    graph = graph_builder.compile(checkpointer=memory)
    try:
        # 增加重试次数和延迟时间
        graph.get_graph().draw_mermaid_png(output_file_path="graph.png", max_retries=5, retry_delay=2.0)
    except Exception as e:
        print(f"渲染图表失败: {e}")
    return graph

def tools():
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
    tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    return [tool, ip_to_city_tool]

def chat(graph, msg):
    # graph = buildGraph()
    message = {"messages": [HumanMessage(content=msg)]}
    for event in graph.stream(message):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# agent架构
"""
定义：agent是使用 LLM 来决定应用程序控制流的系统
1、LLM 可以在两条潜在路径之间进行路由
2、LLM 可以决定调用众多工具中的哪一个
3、LLM 可以决定生成的答案是否足够，或者是否需要更多工作
"""
# 路由器
"""
路由器允许 LLM 从一组指定的选项中选择一个步骤。
这是一种控制级别相对有限的代理架构，
因为 LLM 通常专注于做出单个决策，并从有限的一组预定义选项中生成特定输出
"""
#结构化输出
"""
LLM 的结构化输出通过提供 LLM 在其响应中应遵循的特定格式或模式来工作。
这与工具调用类似，但更通用。虽然工具调用通常涉及选择和使用预定义函数，但结构化输出可用于任何类型的格式化响应。实现结构化输出的常见方法包括：
1、提示工程：通过系统提示指示 LLM 以特定格式响应。
2、输出解析器：使用后处理从 LLM 响应中提取结构化数据。
3、工具调用：利用某些 LLM 内置的工具调用功能生成结构化输出。
"""
#工具调用agent
"""
虽然路由器允许 LLM 做出单个决策，但更复杂的agent架构通过两种关键方式扩展了 LLM 的控制能力：
1、多步决策：LLM 可以连续做出一系列决策，而不仅仅是一个。
2、工具访问：LLM 可以选择并使用各种工具来完成任务。
ReAct 是一种流行的通用agent架构，它结合了这些扩展，并整合了三个核心概念。
1、工具调用：允许 LLM 根据需要选择和使用各种工具。
2、记忆：使代理能够保留和使用来自先前步骤的信息。
3、规划：使 LLM 能够创建并遵循多步计划以实现目标。
这种架构允许更复杂和灵活的代理行为，超越简单的路由，实现多步骤的动态问题解决。与最初的论文不同，今天的代理依赖于 LLM 的工具调用能力，并在一系列消息上运行
"""
#图
"""
LangGraph 的核心是将代理工作流建模为图。使用三个关键组件来定义代理的行为：
1、State：一个共享数据结构，表示应用程序的当前快照。它可以是任何 Python 类型，但通常是 TypedDict 或 Pydantic BaseModel。
2、Nodes：编码代理逻辑的 Python 函数。它们接收当前的 State 作为输入，执行一些计算或副作用，并返回一个更新的 State。
3、Edges：Python 函数，根据当前的 State 决定接下来执行哪个 Node。它们可以是条件分支或固定转换。
LangGraph 的底层图算法使用消息传递来定义通用程序。当一个节点完成其操作时，它会沿着一条或多条边向其他节点发送消息。这些接收节点随后执行其函数，将结果消息传递给下一组节点，然后该过程继续。
超步可以被认为是图节点的一次迭代。并行运行的节点属于同一个超步，而顺序运行的节点属于不同的超步。在图执行开始时，所有节点都处于 inactive 状态。当节点在其任何传入边（或“通道”）上接收到新消息（状态）时，它就会变为 active。活跃节点随后运行其函数并返回更新。在每个超步结束时，没有传入消息的节点通过将自身标记为 inactive 来投票 halt（停止）。当所有节点都处于 inactive 状态且没有消息传输时，图执行终止
"""
#StateGraph
"""
StateGraph 是 LangGraph 中的一个类，用于定义和执行状态图。它允许您将代理的行为表示为一个状态机，其中每个状态都对应于代理的一个特定行为或状态。
"""
# 编译图
"""
要构建图，您首先定义状态，然后添加节点和边，最后编译它。
graph = graph_builder.compile(...)
"""
#状态
"""
定义图时，做的第一件事是定义图的 State。State 包含图的模式以及指定如何应用状态更新的reducer 函数。State 的模式将是图中所有 Nodes 和 Edges 的输入模式，
可以是 TypedDict 或 Pydantic 模型。所有 Nodes 都将发出对 State 的更新，然后使用指定的 reducer 函数应用这些更新
"""
#模式
"""
指定图模式的主要文档化方法是使用 TypedDict。然而，我们也支持使用 Pydantic BaseModel 作为图状态，以添加默认值和额外的数据验证。
"""
#多个模式
"""
通常，所有图节点都使用单个模式进行通信。这意味着它们将读写相同的状态通道。但是，在某些情况下，我们希望对此有更多控制：
1、内部节点可以传递图中输入/输出不需要的信息。
2、我们可能还希望为图使用不同的输入/输出模式。例如，输出可能只包含一个相关的输出键
class InputState(TypedDict):
    user_input: str

class OutputState(TypedDict):
    graph_output: str

class OverallState(TypedDict):
    foo: str
    user_input: str
    graph_output: str

class PrivateState(TypedDict):
    bar: str

def node_1(state: InputState) -> OverallState:
    # Write to OverallState
    return {"foo": state["user_input"] + " name"}

def node_2(state: OverallState) -> PrivateState:
    # Read from OverallState, write to PrivateState
    return {"bar": state["foo"] + " is"}

def node_3(state: PrivateState) -> OutputState:
    # Read from PrivateState, write to OutputState
    return {"graph_output": state["bar"] + " Lance"}

builder = StateGraph(OverallState,input=InputState,output=OutputState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()
graph.invoke({"user_input":"My"})
{'graph_output': 'My name is Lance'}
"""
#归约器:归约器是理解节点更新如何应用于 State 的关键。State 中的每个键都有自己独立的归约函数。
"""
#默认归约器:默认情况下，每个键都使用简单的覆盖归约器。这意味着如果一个节点返回一个值，它将覆盖 State 中该键的当前值。
class State(TypedDict):
    foo: int
    bar: list[str]
#自定义归约器:您可以为每个键指定自定义归约函数。这允许您实现更复杂的更新逻辑，例如合并、累加或转换值。
class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]
    baz: Annotated[str, replace]
#并行执行:在每个超步中，所有节点都并行执行。这意味着节点的更新可以独立应用，而不会相互干扰。
#顺序执行:在每个超步中，节点按照定义的顺序执行。这确保了节点的更新按预期顺序应用。
#图执行:图执行是指在图中执行节点的过程。它从 START 节点开始，沿着定义的边执行，直到到达 END 节点。
#图执行结束:当图执行到达 END 节点时，执行结束。这意味着图的状态已更新，并且可以用于后续处理。
#归约器:归约器是理解节点更新如何应用于 State 的关键。State 中的每个键都有自己独立的归约函数`
"""
#节点
"""
节点是图中的一个基本单位，它表示一个计算步骤或一个状态转换。每个节点都有一个唯一的标识符，用于在图中引用它。
节点类型:LangGraph 支持三种类型的节点：
1、计算节点：执行计算或转换操作的节点。
2、条件节点：根据状态条件执行不同路径的节点。
3、结束节点：表示图执行结束的节点。
节点输入:每个节点都有一个输入，它是一个状态对象，包含图的当前状态。节点可以读取输入状态的键值对，并根据需要进行处理。
节点输出:每个节点都有一个输出，它是一个状态更新对象，包含节点执行后的状态变化。节点可以将输出状态的键值对写入状态对象，以便后续节点使用。
graph_builder.add_node("tools", tool_node)
"""
#START 节点
"""
START 节点是一个特殊节点，代表将用户输入发送到图的节点
graph.add_edge(START, "node_a")
"""
#END 节点
"""
END 节点是一个特殊节点，代表终止节点，图执行结束时到达该节点。
graph.add_edge("node_a", END)
"""
#节点缓存
"""
LangGraph 支持根据节点的输入缓存任务/节点。要使用缓存：
1、编译图时（或指定入口点时）指定缓存。
2、为节点指定缓存策略。每个缓存策略支持：
    （1）key_func 用于根据节点输入生成缓存键，默认为使用 pickle 对输入进行 hash。
    （2）ttl，缓存的存活时间（秒）。如果未指定，缓存将永不过期。
builder.add_node("expensive_node", expensive_node, cache_policy=CachePolicy(ttl=3))
graph = builder.compile(cache=InMemoryCache())
"""
#边
"""
边定义了逻辑如何路由以及图如何决定停止。这是代理工作方式以及不同节点之间如何通信的重要组成部分。边有几种主要类型：
1、普通边：直接从一个节点到下一个节点。
2、条件边：调用函数来确定接下来要到哪个（或哪些）节点。
3、入口点：当用户输入到达时首先调用哪个节点。
4、条件入口点：调用函数来确定当用户输入到达时首先调用哪个（或哪些）节点。
一个节点可以有多个出边。如果一个节点有多个出边，则所有这些目标节点都将作为下一个超步的一部分并行执行。
"""
#普通边
"""
普通边是最简单的边类型，它直接从一个节点到下一个节点。
graph.add_edge("node_a", "node_b")
"""
#条件边
"""
条件边允许根据节点的输入动态确定下一个要调用的节点。这对于实现基于状态的决策逻辑非常有用。
graph.add_conditional_edges(
    "node_a",
    lambda x: "node_b" if x["user_input"] == "hello" else "node_c",
    ["node_b", "node_c"]
)
"""
#入口点
"""
入口点是图启动时运行的第一个节点。您可以使用从虚拟 START 节点到第一个要执行的节点的 add_edge 方法来指定图的入口
graph.a
add_edge(START, "node_a")
"""
#条件入口点
"""
条件入口点允许根据自定义逻辑在不同的节点开始
graph.add_conditional_edges(
    START,
    lambda x: "node_a" if x["user_input"] == "hello" else "node_b",
    ["node_a", "node_b"]
)
"""
#Send
"""
 Send 是一种特殊的边类型，它允许您将状态从一个节点发送到多个节点。
 graph.add_send(
    "node_a",
    ["node_b", "node_c"]
)
"""
#Command
"""
将控制流（边）和状态更新（节点）结合起来会很有用。
例如，在同一个节点中既执行状态更新又决定下一个要去的节点。LangGraph 提供了一种通过从节点函数返回 Command 对象来实现此目的的方法：
from langgraph.graph import Command
def node_a(state: State) -> Command:
    # 执行状态更新
    state["foo"] = 1
    # 决定下一个要去的节点
    return Command(
        next_node="node_b",
        update_state={"bar": ["a"]}
    )
"""
#图迁移
"""
1、即使使用检查点来跟踪状态，LangGraph 也能轻松处理图定义（节点、边和状态）的迁移。
2、对于图末尾的线程（即未中断的线程），您可以更改图的整个拓扑结构（即所有节点和边，删除、添加、重命名等）。
3、对于当前已中断的线程，我们支持除重命名/删除节点之外的所有拓扑更改（因为该线程现在可能即将进入一个不再存在的节点）——如果这是一个障碍，请联系我们，我们可以优先提供解决方案。
4、对于修改状态，我们完全支持添加和删除键的向后和向前兼容性。
5、已重命名的状态键会在现有线程中丢失其保存的状态。
6、类型以不兼容方式更改的状态键目前可能会在具有更改前状态的线程中导致问题——如果这是一个障碍，请联系我们，我们可以优先提供解决方案。
创建图时，您还可以将图的某些部分标记为可配置。这通常是为了方便在模型或系统提示之间切换。这允许您创建一个单一的“认知架构”（图），但拥有它的多个不同实例。
创建图时，您可以选择指定一个 config_schema。
class ConfigSchema(TypedDict):
    llm: str
graph = StateGraph(State, config_schema=ConfigSchema)
"""
# 递归限制¶
"""
递归限制设置了图在单次执行期间可以执行的超步的最大数量。一旦达到限制，LangGraph 将引发 GraphRecursionError。
默认情况下，此值设置为 25 步。递归限制可以在运行时在任何图上设置，
并通过配置字典传递给 .invoke/.stream。重要的是，recursion_limit 是一个独立的 config 键，
不应像所有其他用户定义的配置一样传递到 configurable 键中
graph.invoke(inputs, config={"recursion_limit": 5, "configurable":{"llm": "anthropic"}})
"""
# LangGraph 运行时
"""
Pregel 实现了 LangGraph 的运行时，管理 LangGraph 应用程序的执行。
在 LangGraph 中，Pregel 将执行器（actors）和通道（channels）组合成一个单一的应用程序。执行器从通道读取数据并向通道写入数据。Pregel 按照 Pregel 算法/批量同步并行（Bulk Synchronous Parallel）模型，将应用程序的执行组织成多个步骤。
每个步骤包括三个阶段：
规划：确定在此步骤中要执行哪些执行器。例如，在第一步中，选择订阅特殊输入通道的执行器；在后续步骤中，选择订阅前一步骤中更新的通道的执行器。
执行：并行执行所有选定的执行器，直到全部完成，或一个失败，或达到超时。在此阶段，通道更新对执行器不可见，直到下一个步骤。
更新：用此步骤中执行器写入的值更新通道。
"""
#执行器
"""
一个执行器是一个 PregelNode。它订阅通道，从它们读取数据，并向它们写入数据。它可以被认为是 Pregel 算法中的一个执行器。PregelNodes 实现了 LangChain 的 Runnable 接口。
"""
# 通道
"""
通道用于在执行器（PregelNodes）之间进行通信。每个通道都有一个值类型、一个更新类型和一个更新函数——该函数接受一系列更新并修改存储的值。通道可以用于将数据从一个链发送到另一个链，或在未来的步骤中将数据从一个链发送到自身。LangGraph 提供了许多内置通道：
1、LastValue: 默认通道，存储发送到通道的最后一个值，适用于输入和输出值，或将数据从一个步骤发送到下一个步骤。
2、Topic: 可配置的 PubSub 主题，适用于在执行器之间发送多个值，或用于累积输出。可以配置为重复数据删除或在多个步骤中累积值。
3、BinaryOperatorAggregate: 存储一个持久值，通过对当前值和发送到通道的每个更新应用二元运算符进行更新，适用于在多个步骤中计算聚合
"""
#如何使用图API
#定义状态:LangGraph中的状态可以是TypedDict、Pydantic模型或dataclass
"""
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict
class State(TypedDict):
    messages: list[AnyMessage]
    extra_field: int
"""
#更新状态
"""
from langchain_core.messages import AIMessage


def node(state: State):
    messages = state["messages"]
    new_message = AIMessage("Hello!")
    return {"messages": messages + [new_message], "extra_field": 10}
"""
#使用StateGraph来定义一个在此状态上操作的图
"""
from langgraph.graph import StateGraph
builder = StateGraph(State)
builder.add_node(node)
builder.set_entry_point("node")
graph = builder.compile()
"""
#使用Reducer处理状态更新
"""
状态中的每个键都可以有自己的独立Reducer函数，它控制如何应用来自节点的更新。如果没有明确指定Reducer函数，则假定对键的所有更新都应该覆盖它。
对于TypedDict状态模式，我们可以通过使用Reducer函数注释状态的相应字段来定义Reducer
"""
#定义输入和输出Schema
"""
默认情况下，StateGraph使用单个Schema运行，并且所有节点都应使用该Schema进行通信。但是，也可以为图定义不同的输入和输出Schema
当指定不同Schema时，内部Schema仍将用于节点之间的通信。输入Schema确保提供的输入与预期结构匹配，而输出Schema则过滤内部数据，以根据定义的输出Schema仅返回相关信息
"""
#在节点之间传递私有状态¶
"""
在某些情况下，您可能希望节点交换对中间逻辑至关重要但不需要成为图主Schema一部分的信息。这些私有数据与图的整体输入/输出不相关，应仅在某些节点之间共享
"""
#添加运行时配置
"""
有时您希望在调用图时能够对其进行配置。例如，您可能希望在运行时指定使用哪个LLM或系统提示，而无需用这些参数污染图状态。
要添加运行时配置
指定配置的Schema
将配置添加到节点或条件边的函数签名中
将配置传递给图
def node(state: State, config: RunnableConfig):
    if config["configurable"]["my_runtime_value"] == "a":
        return {"my_state_value": 1}
    elif config["configurable"]["my_runtime_value"] == "b":
        return {"my_state_value": 2}
    else:
        raise ValueError("Unknown values.")
"""
#添加重试策略
"""
在许多用例中，您可能希望节点具有自定义重试策略，例如，如果您正在调用API、查询数据库或调用LLM等。LangGraph允许您向节点添加重试策略。

要配置重试策略，请将retry参数传递给add_node。retry参数接受一个RetryPolicy命名元组对象。下面我们使用默认参数实例化一个RetryPolicy对象并将其与一个节点关联
from langgraph.pregel import RetryPolicy
builder.add_node(
    "node_name",
    node_function,
    retry=RetryPolicy(),
)

"""
#创建分支
"""
节点的并行执行对于加快整体图操作至关重要。LangGraph提供对节点并行执行的原生支持，这可以显著提高基于图的工作流的性能。
这种并行化通过扇出（fan-out）和扇入（fan-in）机制实现，同时利用标准边和条件边。下面是一些示例，展示了如何创建适合您的分支数据流
"""
#并行运行图节点
"""
在此示例中，我们从Node A扇出到B和C，然后扇入到D。通过我们的状态，我们指定Reducer的添加操作。这将组合或累积状态中特定键的值，
而不是简单地覆盖现有值。对于列表，这意味着将新列表与现有列表连接起来。有关使用Reducer更新状态的更多详细信息
"""
#延迟节点执行
"""
当希望延迟节点的执行直到所有其他待处理任务都完成时，延迟节点执行非常有用。这在分支长度不同时尤其重要，这在诸如Map-Reduce流等工作流中很常见
builder.add_node(d, defer=True)
"""
#条件分支





if __name__ == "__main__":
    conf = Config("conf/config.yml")
    model = Model(conf)
    graph = buildGraph()
    # chat("你好")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            chat(graph,user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            chat(graph, user_input)
            break
    # state = {"messages": []}
    # state = graph.invoke(state)
    # print(state)
