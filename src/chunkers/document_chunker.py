from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_document_chunker():
    try:
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        return chunker
    except Exception as e:
        raise RuntimeError(f"Failed to create document chunker: {e}")