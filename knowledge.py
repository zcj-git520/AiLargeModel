"""
知识库 RAG
"""
import logging
from typing import Optional, List

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, Docx2txtLoader, UnstructuredExcelLoader, \
    UnstructuredPowerPointLoader, UnstructuredImageLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from DB.chroma import ChromaDB
from config.config import Config
from enums.model_enums import ModelType

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class Knowledge:
    """文档处理器，负责加载和处理各种类型的文档"""
    def __init__(self):
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,  # 每块最大长度
            chunk_overlap=500,  # 块之间的重叠部分
            length_function=len,
            is_separator_regex=False
        )

        # 文件类型与对应加载器的映射
        self.loader_mapping = {
            "txt": TextLoader,
            "pdf": PyMuPDFLoader,
            "docx": Docx2txtLoader,
            "xlsx": UnstructuredExcelLoader,
            "xls": UnstructuredExcelLoader,
            "pptx": UnstructuredPowerPointLoader,
            "ppt": UnstructuredPowerPointLoader,
            "image": UnstructuredImageLoader,
            "png": UnstructuredImageLoader,
            "jpg": UnstructuredImageLoader,
            "jpeg": UnstructuredImageLoader,
            "default": UnstructuredFileLoader  # 默认加载器
        }

    def clean_text(self, text: str) -> str:
        """清理文本内容"""
        if not text:
            return ""

        # 移除多余的空格和换行
        text = " ".join(text.split())
        # 截断超长文本
        return text[:8000] if len(text) > 8000 else text

    def load_single_file(self, file_path: str, file_type: str) -> Optional[List]:
        """加载单个文件"""
        try:
            # 获取合适的加载器
            loader_class = self.loader_mapping.get(file_type.lower(), self.loader_mapping["default"])

            # 特殊处理：文本文件需要指定编码
            if file_type.lower() == "txt":
                loader = loader_class(file_path, encoding="utf-8")
            else:
                loader = loader_class(file_path)

            return loader.load()
        except Exception as e:
            logger.error(f"加载文件 {file_path} 时出错: {str(e)}")
            return None

    def split_documents(self, documents: List) -> List:
        """分割文档"""
        if not documents:
            return []

        try:
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"分割文档时出错: {str(e)}")
            return documents  # 如果分割失败，返回原始文档


    def load_files(self, file_paths: List[str], file_type: str) -> List:
        """加载并处理多个文件"""
        all_documents = []

        for file_path in file_paths:
            try:
                logger.info(f"正在处理文件: {file_path}")
                documents = self.load_single_file(file_path, file_type)
                if documents:
                    split_docs = self.split_documents(documents)
                    all_documents.extend(split_docs)
            except Exception as e:
                logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
                continue

        return all_documents


if __name__ == "__main__":
    files = ["D:\work\安服工作\安服资料\第2篇：Linux入侵排查.pdf","D:\work\安服工作\安服资料\Web安全开发指南 电子版.pdf"]
    document_processor = Knowledge()
    documents = document_processor.load_files(files, "pdf")
    print(documents)
    #批量写入
    config = Config('conf/config.yml')
    chromadb = ChromaDB(config, ModelType.QWEN)
    chromadb.add(documents)
    # 向量数据库查询
    query = "已监听端⼝ "
    docs = chromadb.query(query)
    for doc in docs:
        print(doc)

