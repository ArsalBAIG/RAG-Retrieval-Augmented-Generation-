from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

persist_dir = './chroma_db'
embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"

db = Chroma(
    persist_directory= persist_dir,
    embedding_function= HuggingFaceEmbeddings(
        model_name= embeddings_model_name),
    collection_metadata={'hnsw:space': 'cosine'}
    )

query = "Nvidia's most significant transformation"
retriever = db.as_retriever(search_kwargs={"k": 3})
relevant_docs = retriever.invoke(query)

print(f'User Query: {query}\n')
for i, doc in enumerate(relevant_docs):
    print(f'Result {i + 1}:')
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f'Content: {doc.page_content[:200]}...')