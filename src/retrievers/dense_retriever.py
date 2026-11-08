from langchain_community.vectorstores import FAISS

from src.chunkers.document_chunker import create_document_chunker
from src.document_load import get_docs
from src.llms.embading_llm import embeddingsLLM


def _create_vectorstore():
    try:
        docs = get_docs()
        docs = create_document_chunker().split_documents(docs)
        vactorstore = FAISS.from_documents(docs, embeddingsLLM)
        return vactorstore
    except Exception as e:
        raise RuntimeError(f"Failed to create vectorstore: {e}")
    
def create_dense_retriever():
    try:
        vectorstore = _create_vectorstore()
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        return dense_retriever
    except Exception as e:
        raise RuntimeError(f"Failed to create dense retriever: {e}")    



