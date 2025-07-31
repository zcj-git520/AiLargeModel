import os
import tempfile
from typing import List, Dict, Any

import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool

from DB.chroma import ChromaDB
from common.modeCommon import Model
from config.config import Config
from enums.model_enums import ModelType
from knowledge import Knowledge


class KnowledgeBaseSystem:
    def __init__(self, model, chromadb: ChromaDB, document_processor: Knowledge):
        self.chromadb = chromadb
        self.model = model
        self.document_processor = document_processor
        self.setup_page_config()
        self.initialize_session()


    def initialize_session(self):
        """初始化会话状态"""
        if "messages" not in st.session_state or st.sidebar.button("清空聊天记录"):
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": "您好，我是您的知识检索助手，请问有什么可以帮助您的？"
                }
            ]

    def setup_page_config(self):
        """设置页面配置"""
        st.set_page_config(
            page_title="智能知识检索系统",
            layout="wide",
            page_icon="📚"
        )
        st.title("📚 智能知识检索系统")
        st.sidebar.header("文档上传")
        # 单独处理清空聊天记录按钮
        if st.sidebar.button("清空聊天记录"):
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": "您好，我是您的知识检索助手，请问有什么可以帮助您的？"
                }
            ]


    def display_chat_history(self):
        """显示聊天历史"""
        # 显示聊天历史
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
        # for msg in st.session_state.messages:
        #     st.chat_message(
        #         msg["role"],
        #     ).write(
        #         msg["content"]
        #     )


    def handle_file_upload(self):
        """处理文件上传"""
        files = st.sidebar.file_uploader(
            label="上传文档（支持txt/pdf/docx/xlsx/pptx/image）",
            type=["txt", "pdf", "docx", "xlsx", "pptx", "image"],
            accept_multiple_files=True,
            help="请上传需要检索的文档"
        )

        if not files:
            return

        for file in files:
            try:
                file_type = file.name.split(".")[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
                    tmp_file.write(file.getvalue())
                    tmp_file_path = tmp_file.name

                with st.spinner(f"正在处理 {file.name}..."):
                    documents = self.document_processor.load_files([tmp_file_path], file_type)
                    self.chromadb.add(documents)
                    st.sidebar.success(f"{file.name} 上传成功")
            except Exception as e:
                st.sidebar.error(f"处理 {file.name} 时出错: {str(e)}")
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

    def create_agent(self) -> AgentExecutor:
        """创建代理执行器"""
        retriever = self.chromadb.as_retriever(search_kwargs={"k": 3})

        tool = create_retriever_tool(
            retriever,
            name="knowledge_retriever",
            description="从知识库中检索相关信息来回答问题"
        )

        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(
            chat_memory=msgs,
            return_messages=True,
            memory_key="chat_history",
            output_key="output"
        )

        prompt_template = """
        您是一个专业的知识检索助手，您的任务是:
        1. 理解用户的问题
        2. 从知识库中检索相关信息
        3. 提供准确、有帮助的回答

        工具:
        ------
        您可以使用的工具:
        {tools}

        使用工具时，请遵循以下格式:
        Thought: 我需要使用工具吗? 是
        Action: 要执行的操作，必须是 [{tool_names}] 中的一个
        Action Input: {input}
        Observations: 操作的结果

        当您有答案要告诉用户，或者不需要使用工具时，必须使用以下格式:
        Thought: 我需要使用工具吗? 否
        Final Answer: [您的回答]

        开始!

        历史对话:
        {chat_history}

        新输入: {input}
        {agent_scratchpad}
        """

        prompt = PromptTemplate.from_template(prompt_template)
        agent = create_react_agent(
            self.model,
            [tool],
            prompt
        )

        return AgentExecutor(
            agent=agent,
            tools=[tool],
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def handle_user_query(self, agent_executor: AgentExecutor):
        """处理用户查询"""
        user_query = st.chat_input(placeholder="请输入您的问题...")

        if not user_query:
            return

        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        # with st.chat_message("assistant"):
        #     st_cb = StreamlitCallbackHandler(st.container())
        #     # print("st_cb: ", st_cb)
        #     config = {
        #         "callbacks": [st_cb]
        #     }
        #     response = agent_executor.invoke(
        #         {
        #             "input": user_query
        #         }, config=config
        #     )
        #     st.session_state.messages.append(
        #         {
        #             "role": "assistant",
        #             "content": response["output"]
        #         }
        #     )
        #     st.write(response["output"])

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container())
            response = agent_executor.invoke(
                {"input": user_query},
                {"callbacks": [st_cb]}
            )

            answer = response.get("output", "抱歉，我无法回答这个问题。")
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.write(answer)

    def run(self):
        """运行知识检索系统"""
        self.display_chat_history()
        self.handle_file_upload()

        try:
            agent_executor = self.create_agent()
            self.handle_user_query(agent_executor)
        except Exception as e:
            st.error(f"系统错误: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "抱歉，系统出现错误，请稍后再试。"
            })


if __name__ == '__main__':
    model_type = ModelType.QWEN
    config = Config("conf/config.yml")
    model = Model(config)
    chromadb = ChromaDB(config, model_type)
    document_processor = Knowledge()
    system = KnowledgeBaseSystem(model.qwen_llm(), chromadb, document_processor)
    system.run()
