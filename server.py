import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_core.runnables import RunnableLambda
from langserve import add_routes

from Prompt import Prompt
from common.modeCommon import Model
from config.config import Config

app = FastAPI(title="LangChain 服务器",
              version="1.0",
              description="使用 Langchain 的 Runnable 接口的简单 API 服务器")


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


def base_prompt_routes(prompt, model):
    base_prompt = prompt.base_prompt()
    llm_chain = model.qwen_llm_china(base_prompt)
    poetry_func = RunnableLambda(lambda x: llm_chain.invoke(x))
    add_routes(app, poetry_func, path="/base")

def chat_prompt_routes(prompt, model):
    chat_prompt = prompt.chat_prompt()
    llm_chain = model.qwen_llm_china(chat_prompt)
    poetry_func = RunnableLambda(lambda x: llm_chain.invoke(x))
    add_routes(app, poetry_func, path="/chat")

def stream_prompt_routes(prompt, model):
    llm_chain = model.qwen_llm_stream()
    poetry_func = RunnableLambda(lambda x: llm_chain.invoke(x))
    add_routes(app, poetry_func, path="/stream")

if __name__ == "__main__":
    # 初始化模型配置
    config = Config('conf/config.yml')
    # 初始化模型
    model = Model(config)
    # # 调用qwen模型
    prompt = Prompt()
    base_prompt_routes(prompt, model)
    chat_prompt_routes(prompt, model)
    stream_prompt_routes(prompt, model)
    uvicorn.run(app, host="0.0.0.0", port=8000)
