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

def extract_person_names(t):
    return {w.lower() for w in re.findall(r"[A-Z][a-z]+", t)}

# =====================================================
# Fix 3: Highlight matching keywords in chunk text
# =====================================================
def highlight_keywords(chunk_text: str, query: str) -> str:
    keywords = [w for w in re.split(r"\W+", query) if len(w) > 2]
    highlighted = chunk_text
    for kw in keywords:
        highlighted = re.sub(
            f"({re.escape(kw)})",
            r"<mark style='background-color:#fff176;padding:0 2px;border-radius:3px;'>\1</mark>",
            highlighted,
            flags=re.IGNORECASE,
        )
    return highlighted

# =====================================================
# URL Loader
# =====================================================
def load_url_as_documents(url):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        t.decompose()
    page_text = "\n".join(
        line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
    )
    return [
        Document(
            page_content=page_text,
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
# Cache embedding model
# =====================================================
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# =====================================================
# Vectorstore helpers
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
# Core ingest function — used by both file and URL ingestion
# =====================================================
def ingest_documents_direct(docs, col_name):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    def make_id(chunk_text, src):
        return hashlib.md5(f"{col_name}:{src}:{chunk_text}".encode()).hexdigest()

    unique = {}
    for c in chunks:
        src = c.metadata.get("source", "")
        uid = make_id(c.page_content, src)
        unique[uid] = c

    vs = get_vectorstore(col_name)
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
    table_name = re.sub(r"[^a-zA-Z0-9_]", "_", os.path.splitext(file.name)[0].lower())
    df.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", c.lower()) for c in df.columns]

    metadata = MetaData()
    columns = [Column("id", Integer, primary_key=True, autoincrement=True)]
    for col, dtype in df.dtypes.items():
        if "int" in str(dtype):
            columns.append(Column(col, BigInteger))
        else:
            columns.append(Column(col, Text))

    table = Table(table_name, metadata, *columns)
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        metadata.create_all(conn)

    df.to_sql(table_name, engine, if_exists="append", index=False, method="multi", chunksize=1000)
    return table_name, df.shape

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

    st.sidebar.subheader("🌐 Add Website URL")
    url_input = st.sidebar.text_input("Enter website URL", placeholder="https://example.com")

    # Fix 2: Ingest URL directly and immediately — no pending state
    if st.sidebar.button("🔗 Ingest URL") and url_input:
        with st.spinner("Fetching and ingesting URL..."):
            try:
                url_docs = load_url_as_documents(url_input)
                count = ingest_documents_direct(url_docs, collection_name)
                st.sidebar.success(f"✅ URL ingested — {count} chunks saved to '{collection_name}'")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Failed to ingest URL: {e}")

else:
    uploaded_tables = st.sidebar.file_uploader(
        "CSV / Excel files", type=["csv", "xlsx"], accept_multiple_files=True
    )

retriever = load_retriever(collection_name)

# Ingest documents button
if st.sidebar.button("📥 Ingest documents"):
    docs = []
    if uploaded_files:
        for f in uploaded_files:
            tmp = f"tmp_{f.name}"
            with open(tmp, "wb") as t:
                t.write(f.read())
            try:
                loader = (
                    PyPDFLoader(tmp) if f.name.endswith(".pdf")
                    else TextLoader(tmp, encoding="utf-8")
                )
                docs.extend(loader.load())
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not load {f.name}: {e}")
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

    if docs:
        with st.spinner("Ingesting documents..."):
            count = ingest_documents_direct(docs, collection_name)
        st.sidebar.success(f"✅ {count} chunks added to '{collection_name}'")
        st.rerun()
    else:
        st.sidebar.warning("⚠️ No files uploaded to ingest.")

# Ingest tables button
if mode == "📊 Database Q&A (Text-to-SQL)" and uploaded_tables:
    if st.sidebar.button("📥 Ingest Tables"):
        for f in uploaded_tables:
            try:
                table, shape = ingest_table(f)
                st.sidebar.success(f"✅ Loaded `{table}` ({shape[0]} rows, {shape[1]} cols)")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to load {f.name}: {e}")

# Clear knowledge base button
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

# Fix 1b: DB count label — contextual based on mode
try:
    doc_count = get_vectorstore(collection_name)._collection.count()
    if mode == "📄 Document Q&A (RAG)":
        st.sidebar.caption(f"📚 Documents in DB: {doc_count}")
    else:
        st.sidebar.caption("💾 Using MySQL database")
except Exception:
    st.sidebar.caption("📚 Documents in DB: —")

# =====================================================
# Session State
# =====================================================
st.session_state.setdefault("session_id", str(uuid4()))
st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_sql_df", None)

# =====================================================
# Fix 1: Visualization panel OUTSIDE chat bubble — persists on chart type change
# =====================================================
if st.session_state.get("last_sql_df") is not None and mode == "📊 Database Q&A (Text-to-SQL)":
    df_vis = st.session_state["last_sql_df"].copy()
    for c in df_vis.columns:
        df_vis[c] = pd.to_numeric(df_vis[c], errors="ignore")

    numeric_cols = df_vis.select_dtypes(include="number").columns.tolist()
    categorical_cols = df_vis.select_dtypes(exclude="number").columns.tolist()

    if numeric_cols and categorical_cols:
        st.subheader("📊 Visualizations")
        col1, col2, col3 = st.columns(3)
        with col1:
            chart_type = st.selectbox("Chart type", ["Bar", "Line", "Area", "Pie"], key="chart_type")
        with col2:
            x_col = st.selectbox("Category (X-axis)", categorical_cols, key="x_col")
        with col3:
            y_col = st.selectbox("Metric (Y-axis)", numeric_cols, key="y_col")

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
        st.divider()

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

                    # Store in session state for visualization panel
                    st.session_state["last_sql_df"] = df_res

                    st.dataframe(df_res, use_container_width=True)

                    answer_summary = f"Query executed successfully. {len(df_res)} rows returned."
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"```sql\n{sql}\n```\n\n{answer_summary}"}
                    )
                    # Rerun so visualization panel at top refreshes
                    st.rerun()

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

                    # Sources
                    sources = list({d.metadata.get("source", "Unknown") for d in retrieved_docs})
                    if sources:
                        st.caption(f"📎 Sources: {', '.join(sources)}")

                    # Fix 3: Top chunks with keyword highlighting
                    with st.expander("🔍 View top matching chunks", expanded=False):
                        for i, doc in enumerate(retrieved_docs[:3]):
                            src = doc.metadata.get("source", "Unknown")
                            highlighted = highlight_keywords(doc.page_content, user_input)
                            st.markdown(
                                f"**Chunk {i+1}** — `{src}`<br>"
                                f"<div style='background:#f8f9fa;padding:10px;border-left:3px solid"
                                f" #4CAF50;border-radius:4px;font-size:0.88em;line-height:1.6'>"
                                f"{highlighted}</div>",
                                unsafe_allow_html=True,
                            )
                            if i < len(retrieved_docs[:3]) - 1:
                                st.divider()

            except Exception as e:
                answer = f"❌ Error: {e}"
                st.error(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})