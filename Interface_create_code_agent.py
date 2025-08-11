# 根据已有的代码，根据用户提供的api，生成对应的代码
import os

from DB.chroma import ChromaDB
from config.config import Config
from enums.model_enums import ModelType
from knowledge import Knowledge

import logging

logger = logging.getLogger(__name__)
def load_file(file_path: list[str], document_processor: Knowledge, chromadb: ChromaDB):
    logger.info("开始加载文件")
    for file_path in file_path:
        data = document_processor.load_files([file_path], "")
        chromadb.add(data)
        logger.info(f"加载文件{file_path}成功")


if __name__ == '__main__':
    file_path = [
        'D:\\project\\xdso-product\\codePecker-SAST-plugin\\src\\main\\java\\com\\shds\\product\\plugin\\CodePeckerSASTManage.java',
        'D:\\project\\xdso-product\\codePecker-SAST-plugin\\src\\main\\java\\com\\shds\\product\\plugin\\CodePeckerSASTPlugin.java',
        'D:\\project\\xdso-product\\codePecker-SAST-plugin\\src\\main\\java\\com\\shds\\product\\plugin\\CodePeckerSASTService.java',
        'D:\\project\\xdso-product\\codePecker-SAST-plugin\\src\\main\\java\\com\\shds\\product\\plugin\\CodeVulnInfo.java',
        'D:\\project\\xdso-product\\codePecker-SAST-plugin\\src\\main\\java\\com\\shds\\product\\plugin\\PagedResponse.java',
        'D:\\project\\xdso-product\\codePecker-SAST-plugin\\src\\main\\java\\com\\shds\\product\\plugin\\CodeVulnInfo.java',
        "D:\\project\\xdso-product\\xdso-product-api\\src\\main\\java\\com\\shds\\product\\domain\\sscs\\CommentReq.java",
        "D:\\project\\xdso-product\\xdso-product-api\\src\\main\\java\\com\\shds\\product\\domain\\sscs\\CreateProjectReq.java",
        "D:\\project\\xdso-product\\xdso-product-api\\src\\main\\java\\com\\shds\\product\\domain\\sscs\\DetectionInfo.java",
        "D:\\project\\xdso-product\\xdso-product-api\\src\\main\\java\\com\\shds\\product\\domain\\sscs\\DetectionReq.java",
        "D:\\project\\xdso-product\\xdso-product-api\\src\\main\\java\\com\\shds\\product\\domain\\sscs\\DetectionRes.java",
        "D:\\project\\xdso-product\\xdso-product-api\\src\\main\\java\\com\\shds\\product\\domain\\sscs\\DetectionStatusRes.java",
        "D:\\project\\xdso-product\\xdso-product-api\\src\\main\\java\\com\\shds\\product\\domain\\sscs\\DownloadReq.java",
        "D:\\project\\xdso-product\\xdso-product-api\\src\\main\\java\\com\\shds\\product\\domain\\sscs\\DownloadReq.java"]
    document_processor = Knowledge()
    config = Config('conf/config.yml')
    chromadb = ChromaDB(config, ModelType.QWEN)
    # load_file(file_path, document_processor,chromadb)
