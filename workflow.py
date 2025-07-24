import asyncio
import logging

from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from langchain.chains import SimpleSequentialChain

import Prompt
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
async def llm_chain(model: Model, promp):
    # 创建一个工作链
    base_prompt = prompt.base_prompt()
    parser = StrOutputParser()
    stream_model = model.qwen_llm_stream()
    chain = base_prompt | stream_model | parser # 创建一个创建一个工作链 先执行base_prompt 然后执行stream_model 最后执行parser
    # 运行工作链
    async for ch in chain.astream(prompt.base_prompt_invoke("系统运维", "运维高级工程师", "想理解交换机原理", "无", "简短文字输出")):
        print(ch, end='|', flush=True)
# 序列链（Sequential Chains）
"""
    序列链是一种将多个链按照顺序连接在一起的工作流。
    每个链都可以接收前一个链的输出作为输入，并将其传递给下一个链。
    序列链可以用于构建复杂的工作流，其中每个链都有特定的任务和逻辑。
    序列链的主要优点是可以将多个链组合在一起，形成一个完整的工作流。
    序列链的主要缺点是每个链都需要独立运行，因此可能会导致性能问题。
"""
def one_chain(model: Model, prompt):
    base_prompt = prompt.base_prompt()
    parser = JsonOutputParser()
    stream_model = model.qwen_llm_stream()
    chain = base_prompt | stream_model | parser
    return chain
def two_chain(model: Model, prompt):
    base_prompt = prompt.base_prompt()
    parser = StrOutputParser()
    stream_model = model.qwen_llm_stream()
    chain = base_prompt | stream_model | parser
    return chain
def llm_chain_sequential(model: Model, prompt):
    chain_one = one_chain(model, prompt)
    chain_two = two_chain(model, prompt)
    overall_chain = chain_one | chain_two
    # 运行工作链
    # 将 prompt.base_prompt_invoke 的返回值包装成字典
    # input_dict =prompt.base_prompt_invoke("系统运维", "运维高级工程师", "想理解交换机原理", "无", "简短文字输出")
    # 运行工作链
    input_dict = {
        "domain": "高铁",
        "profession": "自愿者",
        "do": "去哪里购买高铁票，我应该去找谁",
        "material": "无",
        "question": "按照以下json的格式输出：{"
                    "\"domain\": \"\","
                    "\"profession\": \"\","
                    "\"do\": \"\","
                    "\"material\": \"\", "
                    "\"question\": \"\"}"
    }
    for ch in overall_chain.stream(input_dict):
        print(ch, end='|', flush=True)

#转换链（Transform Chains）： 允许你在 链 的中间步骤中对数据进行转换
def transform_chain(model: Model):
    first_prompt = ChatPromptTemplate.from_template("描述一家生产{产品}的公司最好的名字是什么？")
    chain_one = model.qwen_llm_china(first_prompt)
    second_prompt = ChatPromptTemplate.from_template("为以下公司写一个20字的描述：{company_name}")
    chain_two = model.qwen_llm_china(second_prompt)
    # overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)
    # 直接使用 | 操作符构建序列链
    overall_chain = chain_one | {"company_name": lambda x: x} | chain_two
    product = "外卖和电商"
    input_dict = {"产品": product}
    chain_output = overall_chain.invoke(input_dict)
    print(chain_output)
#条件链（Conditional Chains）： 允许你根据某些条件选择不同的链来执行
#路由链（Router Chains）： 允许你根据某些条件选择不同的链来执行
#多模型链（Multi Model Chains）： 允许你在不同的模型之间切换



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # 初始化模型配置
    config = Config('conf/config.yml')
    # 初始化模型
    model = Model(config)
    # 初始化模板
    prompt = Prompt.Prompt()
    # sync_stream(model, "贵州省的组成")
    # asyncio.run(async_stream(model, "上海怎么样"))
    # asyncio.run(llm_chain(model, prompt))
    # llm_chain_sequential(model, prompt)
    transform_chain(model)