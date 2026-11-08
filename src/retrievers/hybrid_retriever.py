from langchain_classic.retrievers import EnsembleRetriever

from src.retrievers import dense_retriever, sparse_retriever


def create_hybrid_retriever():
    try:
        bm25 = sparse_retriever.create_bm25_retriever()
        dense = dense_retriever.create_dense_retriever()
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25, dense],
            weights=[0.5, 0.5]            
        )                                      
        return hybrid_retriever
    except Exception as e:
        raise RuntimeError(f"Failed to create hybrid retriever: {e}")
