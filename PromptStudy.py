import logging
from langchain.prompts import PromptTemplate

from common.modeCommon import Model
from config.config import Config


# 提示词
def create_prompt(template: str, variables: list) -> PromptTemplate:
    return PromptTemplate(input_variables=variables, template=template)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # 定义提示模板
    template = """
    你是一个专业的{profession}。
    请针对以下问题提供解答：
    {question}
    """
    prompt = create_prompt(variables=["profession", "question"], template=template)
    # 初始化模型配置
    config = Config('conf/config.yml')
    # 初始化模型
    model = Model(config)
    # 调用deepseek模型
    deepseekMode = model.deepseek_llm_china(prompt)
    result = deepseekMode.invoke({"profession": "go工程师", "question": "go的工作原理简短描述"})
    logging.info(result["text"])
    # 调用qwen模型
    chain = model.qwen_llm_china(prompt)
    result = chain.invoke({"profession": "运维工程师", "question": "交换机的工作原理简短描述"})
    logging.info(result["text"])
