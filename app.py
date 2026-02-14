import streamlit as st
from uuid import uuid4
import os, re, hashlib, requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

# RAG chain + memory store
from src.rag_chat_memory import rag_chain_with_memory, store

# Text-to-SQL
from src.text_to_sql.sql_chain import generate_sql
from src.text_to_sql.sql_guard import is_safe_sql
from src.text_to_sql.db import run_sql, engine

# SQLAlchemy imports for schema definition
from sqlalchemy import Table, Column, Integer, Text, BigInteger, MetaData, text

# =====================================================
# Utilities
# =====================================================
def highlight_text(text, query):
    if not query: return text
    keywords = {w.lower() for w in re.findall(r"\w+", query) if len(w) > 2}
    return re.sub(r"\w+", lambda m: f"<mark>{m.group(0)}</mark>" if m.group(0).lower() in keywords else m.group(0), text)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def extract_person_names(text):
    return {w.lower() for w in re.findall(r"[A-Z][a-z]+", text)}

# =====================================================
# URL Loader
# =====================================================
def load_url_as_documents(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "noscript"]): t.decompose()
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    return [Document(page_content=text, metadata={"source": urlparse(url).netloc, "type": "url"})]

# =====================================================
# Streamlit setup
# =====================================================
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chatbot")
st.caption("PDF / TXT / URL → Strict RAG | CSV / XLSX → Text-to-SQL")

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore(collection):
    return Chroma(collection_name=collection, persist_directory="chroma_db", embedding_function=embedding_model)

@st.cache_resource(show_spinner=False)
def load_retriever(collection):
    return get_vectorstore(collection).as_retriever(search_type="similarity", search_kwargs={"k": 6})

# =====================================================
# Sidebar
# =====================================================
st.sidebar.header("🗂️ Collection")
collection_name = st.sidebar.text_input("Collection name", "default")
st.sidebar.divider()
mode = st.sidebar.radio("Chat Mode", ["📄 Document Q&A (RAG)", "📊 Database Q&A (Text-to-SQL)"])

uploaded_files = None
uploaded_tables = None

if mode == "📄 Document Q&A (RAG)":
    uploaded_files = st.sidebar.file_uploader("PDF / TXT files", type=["pdf", "txt"], accept_multiple_files=True)
else:
    uploaded_tables = st.sidebar.file_uploader("CSV / Excel files", type=["csv", "xlsx"], accept_multiple_files=True)

retriever = load_retriever(collection_name)

# =====================================================
# THE FIX: Manual Table Ingestion
# =====================================================
def ingest_table(file):
    try:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        
        # Normalize Names
        table_name = re.sub(r"[^a-zA-Z0-9_]", "_", os.path.splitext(file.name)[0].lower())
        df.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", c.lower()) for c in df.columns]

        metadata = MetaData()
        
        # 1. Define columns for the CREATE TABLE statement
        columns = [Column('id', Integer, primary_key=True, autoincrement=True)]
        
        for col_name, dtype in df.dtypes.items():
            if col_name == 'id': continue # Skip if already in CSV
            if "int" in str(dtype).lower():
                columns.append(Column(col_name, BigInteger))
            elif "float" in str(dtype).lower():
                columns.append(Column(col_name, Text)) # Simple fallback
            else:
                columns.append(Column(col_name, Text))

        # 2. Drop and Recreate table with PRIMARY KEY constraint
        new_table = Table(table_name, metadata, *columns)
        metadata.drop_all(engine, tables=[new_table])
        metadata.create_all(engine)

        # 3. Insert data (Pandas will now append to the existing PK-enabled table)
        df.to_sql(table_name, engine, if_exists="append", index=False, method="multi", chunksize=1000)

        return table_name, df.shape

    except Exception as e:
        st.error(f"❌ Database Ingestion Failed: {str(e)}")
        return None, (0, 0)

# =====================================================
# Logic & Chat (Remaining App Code)
# =====================================================
# ... (Ingest actions, Session State, and Chat Logic from previous version) ...

# =====================================================
# Ingest actions
# =====================================================
if st.sidebar.button("📥 Ingest documents") and uploaded_files:
    docs = []
    for f in uploaded_files:
        tmp = f"tmp_{f.name}"
        with open(tmp, "wb") as t:
            t.write(f.read())
        loader = PyPDFLoader(tmp) if f.name.endswith(".pdf") else TextLoader(tmp)
        docs.extend(loader.load())
        os.remove(tmp)

    ingest_documents(docs)
    st.cache_resource.clear()
    st.sidebar.success("Documents added ✅")
    st.rerun()

if st.sidebar.button("🌍 Ingest URL") and url_input:
    ingest_documents(load_url_as_documents(url_input))
    st.cache_resource.clear()
    st.sidebar.success("Website added ✅")
    st.rerun()

if mode == "📊 Database Q&A (Text-to-SQL)" and uploaded_tables:
    if st.sidebar.button("📥 Ingest Tables"):
        for f in uploaded_tables:
            table, shape = ingest_table(f)
            if table:
                st.sidebar.success(f"Loaded `{table}` ({shape[0]} rows, {shape[1]} cols)")

# =====================================================
# Session & Chat Logic
# =====================================================
st.sidebar.divider()
if st.sidebar.button("🗑️ Clear knowledge base"):
    vs = get_vectorstore(collection_name)
    ids = vs._collection.get().get("ids", [])
    if ids: vs._collection.delete(ids=ids)
    store.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    st.sidebar.success("Cleared ✅")
    st.rerun()

st.session_state.setdefault("session_id", str(uuid4()))
st.session_state.setdefault("messages", [])

if mode == "📄 Document Q&A (RAG)":
    if get_vectorstore(collection_name)._collection.count() == 0:
        st.info("Upload documents or a URL to start.")
        st.stop()

if mode == "📊 Database Q&A (Text-to-SQL)" and not uploaded_tables:
    st.info("Upload CSV or Excel files to start Database Q&A.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if mode == "📊 Database Q&A (Text-to-SQL)":
        with st.chat_message("assistant"):
            sql = generate_sql(user_input)
            if not is_safe_sql(sql):
                answer = "I don't know based on the provided context."
                st.markdown(answer)
            else:
                st.code(sql, language="sql")
                try:
                    rows, cols = run_sql(sql)
                    df_res = pd.DataFrame(rows, columns=cols)
                    st.dataframe(df_res, use_container_width=True)
                    answer = "Here are the results."
                except Exception as e:
                    answer = f"Error running SQL: {e}"
                    st.error(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    else:
        raw_docs = retriever.invoke(user_input)
        docs = []
        seen = set()
        for d in raw_docs:
            t = d.page_content.strip()
            if t and t not in seen:
                seen.add(t); docs.append(d)
            if len(docs) == 3: break

        if not docs:
            answer = "I don't know based on the provided context."
        else:
            context = format_docs(docs)
            if extract_person_names(user_input) - extract_person_names(context):
                answer = "I don't know based on the provided context."
            else:
                answer = rag_chain_with_memory.invoke(
                    {"input": user_input, "context": context},
                    config={"configurable": {"session_id": st.session_state.session_id}},
                )
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})