from langchain_community.document_loaders import DirectoryLoader, TextLoader
from pathlib import Path
import os

data_folder = Path(__file__).parent.parent / "data"
text_loader = TextLoader(str(data_folder / "text_files" / "python_intro.txt"), encoding="utf-8")

try:
    dir_loader = DirectoryLoader(str(data_folder / "text_files"), glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
except Exception as e:
    raise RuntimeError(f"Failed to load documents: {e}")

def get_docs():
    try:
        docs = dir_loader.load()
        return docs
    except Exception as e:
        raise RuntimeError(f"Failed to load documents: {e}")
