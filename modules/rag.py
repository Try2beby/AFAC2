from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core.readers import StringIterableReader
from llama_index.core.node_parser.text import SentenceWindowNodeParser

# import chromadb.utils.embedding_functions as embedding_functions
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

import os
import sys
from loguru import logger

LOG_DIR = "./logs"

# 获取脚本的文件名（不包括扩展名）
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# 使用脚本的文件名作为日志文件的文件名
log_filename = script_name + ".log"

# 设置日志级别
logger.remove()
logger.add(sys.stderr, level="INFO")

# 创建一个处理器，将日志输出到文件
logger.add(os.path.join(LOG_DIR, log_filename), level="INFO")


def resolve_context(idx: int, query: str, raw_context: str):
    logger.info(f"idx: {idx}, query: {query}, raw_context: {raw_context}")

    """
    ====================================================
    切分文本，得到 nodes
    ====================================================
    """
    window_size = 3
    documents = StringIterableReader().load_data(texts=[raw_context])
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    nodes = node_parser.get_nodes_from_documents(documents)

    # len(nodes)
    logger.info(f"#nodes: {len(nodes)}")

    """
    ====================================================
    nodes 入库
    ====================================================
    """
    db = chromadb.PersistentClient(path="./chroma_db")
    # huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    #     model_name="BAAI/bge-m3", api_key=os.getenv("HUGGINGFACE_API_KEY")
    # )

    # create collection
    chroma_collection = db.get_or_create_collection(
        f"bgem3_sentence_test_{idx}_{window_size}",
        metadata={"hnsw:space": "cosine"},
        # embedding_function=huggingface_ef,
    )

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    vector_index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model="local:BAAI/bge-m3",
        show_progress=True,
    )

    """
    ====================================================
    检索
    ====================================================
    """
    retriver = vector_index.as_retriever()
