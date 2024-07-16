from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core.readers import StringIterableReader
from llama_index.core.node_parser.text import SentenceWindowNodeParser
from llama_index.core.node_parser import SentenceSplitter

# import chromadb.utils.embedding_functions as embedding_functions
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from .globals import logger


def resolve_context(idx: int, query: str, raw_context: str, top_k: int = 2):
    logger.info(f"idx: {idx}, query: {query}, raw_context: {raw_context}")

    """
    ====================================================
    切分文本，得到 nodes
    ====================================================
    """
    documents = StringIterableReader().load_data(texts=[raw_context])
    # window_size = 3
    # node_parser = SentenceWindowNodeParser.from_defaults(
    #     window_size=window_size,
    #     window_metadata_key="window",
    #     original_text_metadata_key="original_text",
    # )
    chunk_size = 275
    chunk_overlap = 50
    config_sepc_name = f"{chunk_size}_{chunk_overlap}"
    node_parser = SentenceSplitter(chunk_size=275, chunk_overlap=50)
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
        f"bgem3_sentence_test_{idx}_{config_sepc_name}",
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
    retriever = vector_index.as_retriever(similarity_top_k=top_k)
    retrieved_nodes = retriever.retrieve(query)

    return retrieved_nodes
