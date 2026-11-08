from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
os.environ["HuggingFace_API_key"] = os.getenv("HuggingFace_API_key")

embeddingsLLM = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")