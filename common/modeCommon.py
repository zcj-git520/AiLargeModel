import logging

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from config.config import Config
from enums.model_enums import ModelType


class Model:
    def __init__(self, config: Config):
        self.llm_deepSeek = ChatOpenAI(
            model_name=config.model_config(ModelType.DEEPSEEK).get("modelName"),
            openai_api_key=config.model_config(ModelType.DEEPSEEK).get("apiKey"),
            openai_api_base=config.model_config(ModelType.DEEPSEEK).get("apiBase"),
            max_tokens=config.model_config(ModelType.DEEPSEEK).get("maxTokens"),
        )
        self.llm_qwen = ChatOpenAI(
            model_name=config.model_config(ModelType.QWEN).get("modelName"),
            openai_api_key=config.model_config(ModelType.QWEN).get("apiKey"),
            openai_api_base=config.model_config(ModelType.QWEN).get("apiBase"),
            max_tokens=config.model_config(ModelType.QWEN).get("maxTokens"),
        )

    def llm_china(self, mode_type: ModelType, prompt: PromptTemplate):
        if mode_type == ModelType.DEEPSEEK:
            return LLMChain(llm=self.llm_deepSeek, prompt=prompt)
        if mode_type == ModelType.QWEN:
            return LLMChain(llm=self.llm_qwen, prompt=prompt)
        logging.info(f"modeType: {mode_type} 不存在")
        return None

    def deepseek_llm_china(self, prompt: PromptTemplate):
        return self.llm_china(ModelType.DEEPSEEK, prompt)

    def qwen_llm_china(self, prompt: PromptTemplate):
        return self.llm_china(ModelType.QWEN, prompt)

    # qwen流式输出
    def qwen_llm_stream(self):
        self.llm_qwen.streaming = True
        return self.llm_qwen

    # deepseek流式输出
    def deepseek_qwen_china(self):
        self.llm_deepSeek.streaming = True
        return self.llm_deepSeek
