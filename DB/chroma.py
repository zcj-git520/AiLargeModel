# chroma向量数据库
from chromadb import HttpClient
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

from config.config import Config
from enums.model_enums import ModelType


class ChromaDB:
    def __init__(self, config: Config, model_type: ModelType):
        self.config = config
        try:
            client = HttpClient(host=config.chroma_config().get("host"), port=config.chroma_config().get("port"))

            embedding = DashScopeEmbeddings(
                model=config.model_config(model_type).get("embeddingModelName"),
                dashscope_api_key=config.model_config(model_type).get("apiKey")
            )

            db = Chroma(
                collection_name=config.chroma_config().get("collection_name"),
                client=client,
                embedding_function=embedding  # 传入嵌入函数
            )
            db.get()  # 尝试访问集合验证连接
            print("连接向量数据库成功..")
            self.db = db
        except Exception as e:
            # 异常处理
            raise Exception(f"链接向量数据库失败: {str(e)}, 请检查配置信息") from e

    def add(self, documents):
        # 新增：按API限制拆分批次（每批最多10个文档）
        batch_size = 10
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]  # 截取批次文档
            self.db.add_documents(batch)  # 分批添加
            print(f"已添加第 {i // batch_size + 1} 批文档，共 {len(batch)} 个")
            # 新增：添加文档后，刷新索引

    def query(self, query):
        # 查询文档
        docs = self.db.similarity_search(query)
        return docs

    def delete(self, document_id):
        # 删除文档
        self.db.delete(document_id)
        print(f"已删除文档 ID: {document_id}")

    def update(self, document_id, doc):
        # 更新文档
        self.db.update_document(document_id, doc)
        print(f"已更新文档 ID: {document_id}")

    def list(self):
        # 列出所有文档
        return self.db.get()

    def delete_all(self):
        # 删除所有文档
        self.db.delete_collection()
        print("已删除所有文档")

    def get_id(self, document_id):
        return self.db.get(document_id)

    def as_retriever(self, search_kwargs):
        return self.db.as_retriever(search_kwargs=search_kwargs)

    # def db(self):
    #     return self.db


# if __name__ == '__main__':
#     config = Config('../conf/config.yml')
#     chromadb = ChromaDB(config, ModelType.QWEN)
#     docs = chromadb.list()
#     print(docs)
#     print(chromadb.get_id("0f506577-1db5-42fb-b70d-fe77fb7fe173"))

