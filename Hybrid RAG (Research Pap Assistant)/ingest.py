# --- Importing Lib's ---
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# --- Loading Data & DataBase ---
data_dir = r'C:\Users\ok\OneDrive\Documents\LLMs\RAG(Retrival Augmented Generation)\Hybrid RAG\data'
db_dir = r'C:\Users\ok\OneDrive\Documents\LLMs\RAG(Retrival Augmented Generation)\Hybrid RAG\Database\chroma_db'

# --- PDF/Txt Uploader ---
docs = []
for file in os.listdir(data_dir):
    path = os.path.join(data_dir, file)

    if file.endswith('.pdf'):
        loader = PyPDFLoader(path)
    elif file.endswith('.txt'):
        loader = TextLoader(path)
    else:
        continue
    docs.extend(loader.load()) # Adds to the docs list.

# --- Chunking ---
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size= 800, # reps 800 char's
        chunk_overlap= 100 # reps 100 overlaps.
)
chunks = text_splitter.split_documents(docs)

# --- Embedding & Saving into Database ---
embeddings = HuggingFaceEmbeddings(
    model_name= 'sentence-transformers/all-MiniLM-L6-v2'
)
db = Chroma.from_documents(
    documents= chunks,
    embedding= embeddings,
    persist_directory= db_dir
)
print('Data Indexed Completely!')