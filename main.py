from src.document_load import get_docs
from src.retrievers import hybrid_retriever


def main():
    hybrid = hybrid_retriever.create_hybrid_retriever()
    query = "what is python"
    docs = hybrid.invoke(query)
    for doc in docs:
        print(f"Document: {doc.page_content}\n")
    
    


if __name__ == "__main__":
    main()
