import asyncio
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

from langchain.chains import SimpleSequentialChain

import PromptStudy
from common.modeCommon import Model
from config.config import Config


# LangChain工作流编排

# 同步Stream运行
def sync_stream(model: Model, constr):
    for ch in model.qwen_llm_stream().stream(constr):
        print(ch.content, end='|', flush=True)


# @tool
# 异步AStream运行
async def async_stream(model: Model, constr):
    async for ch in model.qwen_llm_stream().astream(constr):
        print(ch.content, end='|', flush=True)


# Chain
async def llm_chain(model: Model, prompt: PromptStudy):
    # 创建一个工作链
    base_prompt = prompt.base_prompt()
    parser = StrOutputParser()
    stream_model = model.qwen_llm_stream()
    chain = base_prompt | stream_model | parser # 创建一个创建一个工作链 先执行base_prompt 然后执行stream_model 最后执行parser
    # 运行工作链
    async for ch in chain.astream(prompt.base_prompt_invoke("系统运维", "运维高级工程师", "想理解交换机原理", "无", "简短文字输出")):
        print(ch, end='|', flush=True)
# 顺序链：组成多个步骤


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # 初始化模型配置
    config = Config('conf/config.yml')
    # 初始化模型
    model = Model(config)
    # 初始化模板
    prompt = PromptStudy.Prompt()
    # sync_stream(model, "贵州省的组成")
    # asyncio.run(async_stream(model, "上海怎么样"))
    asyncio.run(llm_chain(model, prompt))
