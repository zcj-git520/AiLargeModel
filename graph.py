import json

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, ToolMessage
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
