import logging
from langchain.prompts import PromptTemplate

from common.modeCommon import Model
from config.config import Config

"""提示词模板
提示词模板本质上跟平时大家使用的邮件模板、短信模板没什么区别，就是一个字符串模板，模板可以包含一组模板参数，
通过模板参数值可以替换模板对应的参数。
一个提示词模板可以包含下面内容：
1、发给大语言模型（LLM）的指令。
2、一组问答示例，以提醒AI以什么格式返回请求。
3、发给语言模型的问题。
"""


class Prompt:
    def __init__(self):
        super(Prompt, self).__init__()

    def create_prompt(self, template, variables: list) -> PromptTemplate:
        return PromptTemplate(input_variables=variables, template=template)

    """
    创建基础的模板
    1、你是谁
    2、要干什么事情
    3、干这个事情需要的材料是什么
    4、想要什么样的结构
    """

    def base_prompt(self):
        # 定义提示模板
        template = """
          你是一个{domain}领域的资深{profession}
          需要做一下事情:
          {do}
          给你相关材料以下:
          {material}
          需要返回以下信息：
          {question}
          """
        return self.create_prompt(variables=["domain", "profession", "do", "material", "question"], template=template)

    def base_prompt_invoke(self, domain, profession, do, material, question):
        return {
            "domain": domain,
            "profession": profession,
            "do": do,
            "material": material,
            "question": question,
        }

    """
    聊天模型（Chat Model）以聊天消息列表作为输入，
    这个聊天消息列表的消息内容也可以通过提示词模板进行管理。
    这些聊天消息与原始字符串不同，因为每个消息都与“角色(role)”相关联。
    1、需要给定聊天中角色
    2、给定的聊天内容是什么
    """

    def chat_prompt(self):
        template = """"
        这个一个聊天的场景：
            ("system", "你是一位{role}"),
            ("human", "我是{human_role}"),
            ("human", "{user_input}"),
        """
        return self.create_prompt(variables=["role", "human_role", "user_input"], template=template)

    def chat_prompt_invoke(self, role, human_role, user_input):
        return {
            "role": role,
            "human_role": human_role,
            "user_input": user_input,
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # 初始化模型配置
    config = Config('conf/config.yml')
    # 初始化模型
    model = Model(config)
    # # 调用qwen模型
    prompt = Prompt()
    base_prompt = prompt.base_prompt()
    chain = model.qwen_llm_china(base_prompt)
    result = (chain.invoke(prompt.base_prompt_invoke("系统运维", "运维高级工程师", "想理解交换机原理",
                                                "无", "简短文字输出")))
    print(result["text"])
    chat_prompt = prompt.chat_prompt()
    chain = model.qwen_llm_china(chat_prompt)
    result = chain.invoke(prompt.chat_prompt_invoke("心理学家","心理患者","怎么才能让心情变好，拒绝抑郁呢"))
    print(result["text"])

