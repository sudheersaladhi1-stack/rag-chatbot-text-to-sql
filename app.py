"""Streamlit RAG + Text-to-SQL Chatbot.

Modes:
  📄 Document Q&A  — RAG over PDF / TXT / URL via ChromaDB
  📊 Database Q&A  — Natural-language → SQL over CSV / XLSX loaded into MySQL
"""

import hashlib
import os
import re
import tempfile

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sqlalchemy import Column, Integer, Text, BigInteger, MetaData, Table, text
from urllib.parse import urlparse
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# RAG
from src.rag_chat_memory import rag_chain_with_memory, store

# Text-to-SQL
from src.text_to_sql.db import engine, run_sql
from src.text_to_sql.schema_loader import get_schema
from src.text_to_sql.sql_chain import generate_sql
from src.text_to_sql.sql_guard import is_safe_sql

# =====================================================
# BUG 1 FIX: Greeting detection
# Previously: greetings fell into generate_sql() which
# produced queries like: SELECT * FROM forecast WHERE
# customer_name = 'HI'  →  0 rows, blank response.
# Now: intercept before SQL/RAG and reply helpfully.
# =====================================================
_GREETINGS = {
    "hi", "hello", "hey", "hiya", "howdy", "sup", "yo",
    "good morning", "good afternoon", "good evening",
    "what's up", "whats up", "greetings",
}

def is_greeting(text: str) -> bool:
    return text.strip().lower().rstrip("!.,?") in _GREETINGS


def greeting_response(mode: str) -> str:
    if "Document" in mode:
        return (
            "👋 Hello! I'm your **Document Q&A assistant**.\n\n"
            "To get started:\n"
            "1. **Upload** a PDF or TXT file in the sidebar.\n"
            "2. Click **Ingest documents** to index it.\n"
            "3. Then ask me anything about the content!"
        )
    else:
        return (
            "👋 Hello! I'm your **Database Q&A assistant**.\n\n"
            "To get started:\n"
            "1. **Upload** a CSV or Excel file in the sidebar.\n"
            "2. Click **Ingest Tables** to load it into the database.\n"
            "3. Then ask questions like *'total forecast by market'* and "
            "I'll write the SQL and show you the results with a chart!"
        )


# =====================================================
# Utilities
# =====================================================
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def load_url_as_documents(url: str) -> list[Document]:
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text_content = "\n".join(
        line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
    )
    return [
        Document(
            page_content=text_content,
            metadata={"source": urlparse(url).netloc, "type": "url"},
        )
    ]


# =====================================================
# Streamlit setup
# =====================================================
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chatbot with Visual Analytics")
st.caption("PDF / TXT / URL → Strict RAG  |  CSV / XLSX → Text-to-SQL")


# =====================================================
# Vector Store (cached)
# =====================================================
@st.cache_resource(show_spinner=False)
def get_embedding_model() -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def get_vectorstore(collection: str) -> Chroma:
    return Chroma(
        collection_name=collection,
        persist_directory="chroma_db",
        embedding_function=get_embedding_model(),
    )


@st.cache_resource(show_spinner=False)
def load_retriever(collection: str):
    return get_vectorstore(collection).as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )


# =====================================================
# BUG 3 FIX: DB table stats in sidebar
# Previously: sidebar only showed "Using MySQL database"
# with no information about what tables/rows are loaded.
# Now: queries COUNT(*) for each table and displays it.
# =====================================================
def show_db_stats() -> None:
    try:
        schema = get_schema()
        if not schema:
            st.sidebar.caption("📭 No tables loaded yet.")
            return
        st.sidebar.markdown("**📋 Tables in database:**")
        for table_name in schema:
            try:
                with engine.connect() as conn:
                    row = conn.execute(
                        text(f"SELECT COUNT(*) FROM `{table_name}`")
                    ).fetchone()
                    count = row[0] if row else 0
                st.sidebar.caption(f"• `{table_name}` — **{count:,} records**")
            except Exception:
                st.sidebar.caption(f"• `{table_name}` — (count unavailable)")
    except Exception as e:
        st.sidebar.caption(f"⚠️ Could not load DB info: {e}")


# =====================================================
# Sidebar — Collection & Mode
# =====================================================
st.sidebar.header("🗂️ Collection")
collection_name = st.sidebar.text_input("Collection name", "default")

mode = st.sidebar.radio(
    "Chat Mode",
    ["📄 Document Q&A (RAG)", "📊 Database Q&A (Text-to-SQL)"],
)

uploaded_files = None
uploaded_tables = None

if mode == "📄 Document Q&A (RAG)":
    st.sidebar.header("📂 Upload Files")
    uploaded_files = st.sidebar.file_uploader(
        "PDF / TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )
else:
    st.sidebar.header("📊 CSV / Excel files")
    uploaded_tables = st.sidebar.file_uploader(
        "CSV / Excel files", type=["csv", "xlsx"], accept_multiple_files=True
    )
    st.sidebar.divider()
    st.sidebar.caption("🗄️ Using MySQL database")
    show_db_stats()  # BUG 3 FIX: show table + record counts


# =====================================================
# Ingest Documents
# =====================================================
def ingest_documents(docs: list[Document], collection: str) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    unique: dict[str, Document] = {}
    for c in chunks:
        src = c.metadata.get("source", "")
        uid = hashlib.md5(f"{collection}:{src}:{c.page_content}".encode()).hexdigest()
        unique[uid] = c
    vs = get_vectorstore(collection)
    vs.add_documents(list(unique.values()), ids=list(unique.keys()))


# =====================================================
# Ingest Tables (PK-safe for MySQL)
# =====================================================
def ingest_table(file) -> tuple[str, tuple]:
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
        columns.append(Column(col, BigInteger if "int" in str(dtype) else Text))

    Table(table_name, metadata, *columns)
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))
        metadata.create_all(conn)

    df.to_sql(table_name, engine, if_exists="append", index=False, method="multi", chunksize=1000)
    return table_name, df.shape


# =====================================================
# Sidebar Actions
# =====================================================
if mode == "📄 Document Q&A (RAG)":
    if st.sidebar.button("📥 Ingest documents"):
        if not uploaded_files:
            st.sidebar.warning("Upload at least one file first.")
        else:
            docs: list[Document] = []
            for f in uploaded_files:
                suffix = ".pdf" if f.name.endswith(".pdf") else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                try:
                    loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path)
                    docs.extend(loader.load())
                finally:
                    os.remove(tmp_path)
            ingest_documents(docs, collection_name)
            st.sidebar.success("Documents ingested ✅")
            st.rerun()

    st.sidebar.header("🌐 Add Website URL")
    url_input = st.sidebar.text_input("Enter website URL")
    if st.sidebar.button("🌍 Ingest URL"):
        if not url_input:
            st.sidebar.warning("Enter a valid URL.")
        else:
            ingest_documents(load_url_as_documents(url_input), collection_name)
            st.sidebar.success("Website ingested ✅")
            st.rerun()

else:
    if st.sidebar.button("📥 Ingest Tables"):
        if not uploaded_tables:
            st.sidebar.warning("Upload at least one CSV or Excel file first.")
        else:
            for f in uploaded_tables:
                try:
                    table, shape = ingest_table(f)
                    st.sidebar.success(
                        f"Loaded `{table}` ({shape[0]:,} rows, {shape[1]} cols) ✅"
                    )
                except Exception as e:
                    st.sidebar.error(f"Failed to ingest `{f.name}`: {e}")
            st.rerun()

st.sidebar.divider()
if st.sidebar.button("🗑️ Clear knowledge base"):
    vs = get_vectorstore(collection_name)
    ids = vs._collection.get().get("ids", [])
    if ids:
        vs._collection.delete(ids=ids)
    store.clear()
    st.session_state.clear()
    st.sidebar.success("Knowledge base cleared ✅")
    st.rerun()

if mode == "📄 Document Q&A (RAG)":
    doc_count = get_vectorstore(collection_name)._collection.count()
    st.sidebar.caption(f"📄 Documents in DB: {doc_count}")


# =====================================================
# Session State
# =====================================================
st.session_state.setdefault("session_id", str(uuid4()))
st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_sql_df", None)
st.session_state.setdefault("last_sql", "")


# =====================================================
# BUG 2 FIX: Visualization renderer
#
# Previously: st.selectbox + st.plotly_chart were INSIDE
# `with st.chat_message("assistant"):` — Streamlit discards
# widgets inside chat bubbles on rerun, so the chart
# disappeared the moment the user interacted with it.
#
# Now: results/charts are rendered OUTSIDE all chat bubbles,
# at the top of the page, persisted in session_state so they
# survive any rerun (including selectbox interactions).
# =====================================================
def render_visualization(df_res: pd.DataFrame, sql: str) -> None:
    st.markdown("---")
    st.subheader("📊 Query Results")
    st.code(sql, language="sql")
    st.dataframe(df_res, use_container_width=True)
    st.caption(f"✅ {len(df_res):,} rows returned.")

    df_vis = df_res.copy()
    for c in df_vis.columns:
        df_vis[c] = pd.to_numeric(df_vis[c], errors="ignore")

    numeric_cols = df_vis.select_dtypes(include="number").columns.tolist()
    categorical_cols = df_vis.select_dtypes(exclude="number").columns.tolist()

    if numeric_cols and categorical_cols:
        st.subheader("📈 Visualizations")
        col1, col2, col3 = st.columns(3)
        with col1:
            chart_type = st.selectbox(
                "Chart type", ["Bar", "Line", "Area", "Pie"], key="chart_type"
            )
        with col2:
            x_col = st.selectbox("Category (X)", categorical_cols, key="x_col")
        with col3:
            y_col = st.selectbox("Metric (Y)", numeric_cols, key="y_col")

        if chart_type == "Bar":
            fig = px.bar(df_vis, x=x_col, y=y_col, text=y_col)
        elif chart_type == "Line":
            fig = px.line(df_vis, x=x_col, y=y_col, markers=True)
        elif chart_type == "Area":
            fig = px.area(df_vis, x=x_col, y=y_col)
        else:
            fig = px.pie(df_vis, names=x_col, values=y_col, hole=0.35)

        fig.update_layout(margin=dict(t=40, l=20, r=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    elif len(df_res) > 0:
        st.info("ℹ️ Table-only result — no numeric+categorical pair available for charting.")

    st.markdown("---")


# Render persisted SQL results at the top (survives all reruns)
if (
    mode == "📊 Database Q&A (Text-to-SQL)"
    and st.session_state["last_sql_df"] is not None
):
    render_visualization(
        st.session_state["last_sql_df"],
        st.session_state["last_sql"],
    )


# =====================================================
# Chat history
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

    # ── BUG 1 FIX: Catch greetings before SQL/RAG ─────────────────────────
    if is_greeting(user_input):
        answer = greeting_response(mode)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.stop()

    # ── TEXT-TO-SQL ────────────────────────────────────────────────────────
    if mode == "📊 Database Q&A (Text-to-SQL)":
        sql = generate_sql(user_input)

        if not is_safe_sql(sql):
            answer = "⚠️ I can only run SELECT queries. Please rephrase your question."
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        else:
            try:
                rows, cols = run_sql(sql)
                df_res = pd.DataFrame(rows, columns=list(cols))

                # BUG 2 FIX: persist so render_visualization() above survives reruns
                st.session_state["last_sql_df"] = df_res
                st.session_state["last_sql"] = sql

                # Only a short summary goes into the chat bubble
                summary = f"✅ Query executed. **{len(df_res):,} rows** returned. See results above ↑"
                st.session_state.messages.append({"role": "assistant", "content": summary})
                with st.chat_message("assistant"):
                    st.markdown(summary)

            except Exception as e:
                answer = f"❌ SQL Execution Error: {e}"
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.error(answer)

        st.rerun()  # triggers render_visualization() at top of page

    # ── RAG ────────────────────────────────────────────────────────────────
    else:
        retriever = load_retriever(collection_name)
        raw_docs: list[Document] = []
        try:
            raw_docs = retriever.invoke(user_input)
        except Exception as e:
            st.warning(f"Retrieval failed: {e}")

        seen: set[str] = set()
        docs: list[Document] = []
        for d in raw_docs:
            t = d.page_content.strip()
            if t and t not in seen:
                seen.add(t)
                docs.append(d)
            if len(docs) == 3:
                break

        if not docs:
            answer = "I don't know based on the provided context."
        else:
            context = format_docs(docs)
            answer = rag_chain_with_memory.invoke(
                {"input": user_input, "context": context},
                config={"configurable": {"session_id": st.session_state.session_id}},
            )

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)