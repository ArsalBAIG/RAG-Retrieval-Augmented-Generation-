# --- Loading Lib's ---
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- Doing Directory-Loading ---
db_dir = r'C:\Users\ok\OneDrive\Documents\LLMs\RAG(Retrival Augmented Generation)\Hybrid RAG\Database\chroma_db'
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'}  # Or 'cuda' if you have a GPU
)

# --- Adding New File Function & Loading & Chunking & Embedding.---
def add_new_files(uploaded_files):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    new_chunks = []
    
    for uploaded_file in uploaded_files:
# suffix will extract the extension.
        with tempfile.NamedTemporaryFile(delete=False, suffix= os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue()) # write bytes of data on disk.
            path = tmp.name
        
        loader = PyPDFLoader(path) if path.endswith('.pdf') else TextLoader(path)
        new_chunks.extend(splitter.split_documents(loader.load()))
        os.remove(path)
    # Adds to the existing db_dir without deleting old data
    Chroma.from_documents(
        documents=new_chunks,
        embedding=embeddings,
        persist_directory=db_dir)

# --- App UI Setup ---
st.set_page_config(page_title='Research Paper Assistant!')
st.title('📚 Research Paper Assistant (Hybrid RAG)')

# --- Sidebar Uploader ---
with st.sidebar:
    st.header("Add New Papers")
    new_files = st.file_uploader("Upload PDF/TXT", type=['pdf', 'txt'], accept_multiple_files=True)
    if st.button("Add Files"):
        if new_files:
            with st.spinner("Processing..."):
                add_new_files(new_files)
                st.success("Database Updated!")
                st.rerun()

# --- Search & LLM Logic ---
vector_db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
all_docs = vector_db.get()['documents']

if not all_docs:
    st.info("Database is empty. Use the sidebar to upload.")
    st.stop()

# bm25 stores results as pair of (text, score)
bm25 = BM25Okapi([doc.lower().split() for doc in all_docs])
llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0, groq_api_key= os.getenv('GROQ_API_KEY'))

# --- User Query ---
def ask_question(query):
    # Hybrid Search
    vec_res = vector_db.similarity_search(query, k=3)

# split the query then get their scores then pair each doc with it's score and sorted it then took the x[1] which rep score and performs in acending order.
    bm25_res = sorted(zip(all_docs, bm25.get_scores(query.lower().split())), key=lambda x: x[1], reverse=True)[:3]

# It combines raw text from 'vec_res' list and 'bm25_res' list.
    context = "\n".join(list(set([d.page_content for d in vec_res] + [text for text, score in bm25_res]))) # set is use for de-duplication.
    prompt = f"Context: {context} \n Question: {query}"
    return llm.invoke([HumanMessage(content=prompt)]).content

# --- Streamlit Interaction ---
query = st.text_input('Ask a question about your papers:')
if st.button('Ask') and query:
    with st.spinner('Thinking...'):
        answer = ask_question(query)
        st.markdown('### 📄 Answer')
        st.write(answer)