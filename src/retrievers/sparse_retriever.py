from langchain_community.retrievers import BM25Retriever
from src.document_load import get_docs
from src.chunkers.document_chunker import create_document_chunker
def create_bm25_retriever():
    try:
        docs = get_docs()
        docs = create_document_chunker().split_documents(docs)
        
        bm25_retriever = BM25Retriever.from_documents(docs)
        return bm25_retriever
    except Exception as e:
        raise RuntimeError(f"Failed to create BM25 retriever: {e}")

