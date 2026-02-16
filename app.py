import streamlit as st
from uuid import uuid4
import os, re, hashlib, requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import pandas as pd
import plotly.express as px

from sqlalchemy import (
    Table, Column, Integer, Text, BigInteger, MetaData, text
)

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

# RAG
from src.rag_chat_memory import rag_chain_with_memory, store

# Text-to-SQL
from src.text_to_sql.sql_chain import generate_sql
from src.text_to_sql.sql_guard import is_safe_sql
from src.text_to_sql.db import run_sql, engine

CHROMA_DIR = "chroma_db"

# =====================================================
# Utilities
# =====================================================
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
    for t in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        t.decompose()
    text = "\n".join(
        line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
    )
    return [
        Document(
            page_content=text,
            metadata={"source": urlparse(url).netloc, "type": "url"},
        )
    ]

# =====================================================
# Streamlit setup
# =====================================================
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chatbot with Visual Analytics")
st.caption("PDF / TXT / URL → Strict RAG | CSV / XLSX → Text-to-SQL")

# =====================================================
# ✅ Fixed: Cache embedding model — prevents recreation on every rerun
# =====================================================
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# =====================================================
# ✅ Fixed: Cache vectorstore retriever per collection
# =====================================================
def get_vectorstore(collection):
    return Chroma(
        collection_name=collection,
        persist_directory=CHROMA_DIR,
        embedding_function=get_embedding_model(),
    )

@st.cache_resource(show_spinner=False)
def load_retriever(collection):
    return get_vectorstore(collection).as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )

# =====================================================
# Sidebar
# =====================================================
st.sidebar.header("🗂️ Collection")
collection_name = st.sidebar.text_input("Collection name", "default")

mode = st.sidebar.radio(
    "Chat Mode",
    ["📄 Document Q&A (RAG)", "📊 Database Q&A (Text-to-SQL)"]
)

uploaded_files = None
uploaded_tables = None

if mode == "📄 Document Q&A (RAG)":
    st.sidebar.subheader("📁 Upload Files")
    uploaded_files = st.sidebar.file_uploader(
        "PDF / TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )

    # ✅ Fixed: URL ingestion UI was coded but never shown — now wired to sidebar
    st.sidebar.subheader("🌐 Add Website URL")
    url_input = st.sidebar.text_input("Enter website URL", placeholder="https://example.com")

    if st.sidebar.button("🔗 Ingest URL") and url_input:
        with st.sidebar:
            with st.spinner("Fetching URL..."):
                try:
                    docs = load_url_as_documents(url_input)
                    ingest_documents_fn = lambda d: None  # defined below, called after
                    st.session_state["_pending_url_docs"] = docs
                    st.sidebar.success(f"URL fetched ✅ — click 'Ingest documents' to save.")
                except Exception as e:
                    st.sidebar.error(f"Failed to fetch URL: {e}")

else:
    uploaded_tables = st.sidebar.file_uploader(
        "CSV / Excel files", type=["csv", "xlsx"], accept_multiple_files=True
    )

retriever = load_retriever(collection_name)

# =====================================================
# Ingest Documents
# =====================================================
def ingest_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    def make_id(chunk_text, src):
        return hashlib.md5(
            f"{collection_name}:{src}:{chunk_text}".encode()
        ).hexdigest()

    unique = {}
    for c in chunks:
        src = c.metadata.get("source", "")
        uid = make_id(c.page_content, src)
        unique[uid] = c

    vs = get_vectorstore(collection_name)
    vs.add_documents(list(unique.values()), ids=list(unique.keys()))
    return len(unique)

# =====================================================
# Ingest Tables (PK-safe for MySQL)
# =====================================================
def ingest_table(file):
    df = (
        pd.read_csv(file)
        if file.name.endswith(".csv")
        else pd.read_excel(file, engine="openpyxl")
    )

    table_name = re.sub(
        r"[^a-zA-Z0-9_]",
        "_",
        os.path.splitext(file.name)[0].lower(),
    )
    df.columns = [
        re.sub(r"[^a-zA-Z0-9_]", "_", c.lower())
        for c in df.columns
    ]

    metadata = MetaData()
    columns = [
        Column("id", Integer, primary_key=True, autoincrement=True)
    ]

    for col, dtype in df.dtypes.items():
        if "int" in str(dtype):
            columns.append(Column(col, BigInteger))
        else:
            columns.append(Column(col, Text))

    table = Table(table_name, metadata, *columns)

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        metadata.create_all(conn)

    df.to_sql(
        table_name,
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )

    return table_name, df.shape

# =====================================================
# Sidebar Actions
# =====================================================
if st.sidebar.button("📥 Ingest documents"):
    docs = []

    # Ingest uploaded files
    if uploaded_files:
        for f in uploaded_files:
            tmp = f"tmp_{f.name}"
            with open(tmp, "wb") as t:
                t.write(f.read())
            try:
                loader = (
                    PyPDFLoader(tmp)
                    if f.name.endswith(".pdf")
                    else TextLoader(tmp, encoding="utf-8")
                )
                docs.extend(loader.load())
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not load {f.name}: {e}")
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

    # ✅ Also ingest any pending URL docs
    if "_pending_url_docs" in st.session_state:
        docs.extend(st.session_state.pop("_pending_url_docs"))

    if docs:
        with st.spinner("Ingesting documents..."):
            count = ingest_documents(docs)
        st.sidebar.success(f"✅ {count} chunks added to '{collection_name}'")
        st.rerun()
    else:
        st.sidebar.warning("⚠️ No documents or URLs to ingest.")

if mode == "📊 Database Q&A (Text-to-SQL)" and uploaded_tables:
    if st.sidebar.button("📥 Ingest Tables"):
        for f in uploaded_tables:
            try:
                table, shape = ingest_table(f)
                st.sidebar.success(
                    f"✅ Loaded `{table}` ({shape[0]} rows, {shape[1]} cols)"
                )
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load {f.name}: {e}")

# =====================================================
# Clear Knowledge Base
# =====================================================
if st.sidebar.button("🗑️ Clear knowledge base"):
    try:
        vs = get_vectorstore(collection_name)
        ids = vs._collection.get().get("ids", [])
        if ids:
            vs._collection.delete(ids=ids)
            st.sidebar.success(f"✅ Cleared {len(ids)} chunks from '{collection_name}'")
        else:
            st.sidebar.info("Collection is already empty.")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"❌ Error clearing knowledge base: {e}")

# Show document count
try:
    doc_count = get_vectorstore(collection_name)._collection.count()
    st.sidebar.caption(f"📚 Documents in DB: {doc_count}")
except Exception:
    st.sidebar.caption("📚 Documents in DB: —")

# =====================================================
# Session State
# =====================================================
st.session_state.setdefault("session_id", str(uuid4()))
st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_sql_df", None)

# =====================================================
# Chat History
# =====================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =====================================================
# Chat Input
# =====================================================
user_input = st.chat_input("Ask a question based on the uploaded knowledge")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # ================= TEXT-TO-SQL =================
    if mode == "📊 Database Q&A (Text-to-SQL)":
        with st.chat_message("assistant"):
            sql = generate_sql(user_input)

            if not is_safe_sql(sql):
                answer = "⚠️ I can only run SELECT queries. Please rephrase your question."
                st.warning(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            else:
                st.code(sql, language="sql")

                try:
                    rows, cols = run_sql(sql)
                    df_res = pd.DataFrame(rows, columns=cols)

                    st.session_state["last_sql_df"] = df_res
                    st.session_state["last_sql"] = sql

                    st.dataframe(df_res, use_container_width=True)

                    # ✅ Fixed: Save successful SQL answer to chat history
                    answer_summary = f"Query executed successfully. {len(df_res)} rows returned."
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"```sql\n{sql}\n```\n\n{answer_summary}"}
                    )

                    # =====================================================
                    # 📊 Visualization Panel
                    # =====================================================
                    st.subheader("📊 Visualizations")

                    df_vis = df_res.copy()
                    for c in df_vis.columns:
                        df_vis[c] = pd.to_numeric(df_vis[c], errors="ignore")

                    numeric_cols = df_vis.select_dtypes(include="number").columns.tolist()
                    categorical_cols = df_vis.select_dtypes(exclude="number").columns.tolist()

                    if numeric_cols and categorical_cols:
                        chart_type = st.selectbox("Chart type", ["Bar", "Line", "Area", "Pie"])
                        x_col = st.selectbox("Category (X-axis)", categorical_cols)
                        y_col = st.selectbox("Metric (Y-axis)", numeric_cols)

                        if chart_type == "Bar":
                            fig = px.bar(df_vis, x=x_col, y=y_col, text=y_col)
                        elif chart_type == "Line":
                            fig = px.line(df_vis, x=x_col, y=y_col, markers=True)
                        elif chart_type == "Area":
                            fig = px.area(df_vis, x=x_col, y=y_col)
                        elif chart_type == "Pie":
                            fig = px.pie(df_vis, names=x_col, values=y_col, hole=0.35)

                        fig.update_layout(margin=dict(t=40, l=20, r=20, b=20))
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.info("ℹ️ Not enough numeric and categorical columns for visualization.")

                except Exception as e:
                    answer = f"❌ SQL Execution Error: {e}"
                    st.error(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    # ================= RAG Q&A =================
    else:
        with st.chat_message("assistant"):
            try:
                retrieved_docs = retriever.invoke(user_input)
                context = format_docs(retrieved_docs)

                if not context.strip():
                    answer = "I don't know based on the provided context."
                    st.markdown(answer)
                else:
                    with st.spinner("Thinking..."):
                        answer = rag_chain_with_memory.invoke(
                            {"input": user_input, "context": context},
                            config={"configurable": {"session_id": st.session_state["session_id"]}},
                        )
                    st.markdown(answer)

                    # Show sources
                    sources = list({
                        d.metadata.get("source", "Unknown") for d in retrieved_docs
                    })
                    if sources:
                        st.caption(f"📎 Sources: {', '.join(sources)}")

            except Exception as e:
                answer = f"❌ Error: {e}"
                st.error(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})