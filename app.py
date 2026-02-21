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
from src.text_to_sql.schema_loader import get_schema, get_schema_legacy
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


def highlight_text(text: str, query: str) -> str:
    """Highlight query words in text for the chunks debug panel."""
    import html
    text = html.escape(text)
    words = re.findall(r"\w+", query.lower())
    for word in set(words):
        if len(word) < 3:
            continue
        pattern = re.compile(rf"({re.escape(word)})", re.IGNORECASE)
        text = pattern.sub(
            r"<mark style='background-color:#ffe066'>\1</mark>",
            text,
        )
    return text


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
        schema = get_schema_legacy()  # Use simple format for sidebar
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
    # Clear RAG vector DB
    vs = get_vectorstore(collection_name)
    ids = vs._collection.get().get("ids", [])
    if ids:
        vs._collection.delete(ids=ids)
    
    # FIX Issue 1: Also clear ALL SQL tables
    try:
        schema = get_schema_legacy()  # Use simple format
        if schema:
            with engine.begin() as conn:
                for table_name in schema:
                    conn.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))
    except Exception as e:
        st.sidebar.warning(f"Could not clear SQL tables: {e}")
    
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
# BUG 2 FIX: Visualization renderer with unique keys
# Each message needs unique widget keys to avoid
# StreamlitDuplicateElementKey when rendering multiple
# SQL results in chat history.
# =====================================================
def render_visualization(df_res: pd.DataFrame, sql: str, msg_index: int, insights: str = "") -> None:
    st.markdown("---")
    st.subheader("📊 Query Results")
    st.code(sql, language="sql")
    
    # Reset index to start from 1 instead of 0
    df_display = df_res.copy()
    df_display.index = range(1, len(df_display) + 1)
    
    st.dataframe(df_display, use_container_width=True)
    st.caption(f"✅ {len(df_res):,} rows returned.")
    
    # Show insights if available
    if insights:
        st.info(f"💡 **Key Insights:** {insights}")

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
                "Chart type", 
                ["Bar", "Line", "Area", "Pie"], 
                key=f"chart_type_{msg_index}"
            )
        with col2:
            x_col = st.selectbox(
                "Category (X)", 
                categorical_cols, 
                key=f"x_col_{msg_index}"
            )
        with col3:
            y_col = st.selectbox(
                "Metric (Y)", 
                numeric_cols, 
                key=f"y_col_{msg_index}"
            )

        if chart_type == "Bar":
            fig = px.bar(df_vis, x=x_col, y=y_col, text=y_col)
        elif chart_type == "Line":
            fig = px.line(df_vis, x=x_col, y=y_col, markers=True)
        elif chart_type == "Area":
            fig = px.area(df_vis, x=x_col, y=y_col)
        else:
            fig = px.pie(df_vis, names=x_col, values=y_col, hole=0.35)

        fig.update_layout(margin=dict(t=40, l=20, r=20, b=20))
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{msg_index}")
    elif len(df_res) > 0:
        st.info("ℹ️ Table-only result — no numeric+categorical pair available for charting.")

    st.markdown("---")


# =====================================================
# Chat history
# FIX Issue 2: SQL results now render INLINE with their
# respective question in chat history, not at the top.
# =====================================================
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Thinking expander — collapsed in history
        if msg["role"] == "assistant" and msg.get("thinking"):
            with st.expander("🧠 View AI thinking process", expanded=False):
                st.markdown(msg["thinking"])

        # Unsafe SQL debug
        if msg["role"] == "assistant" and msg.get("unsafe_sql"):
            with st.expander("🔍 See generated SQL"):
                st.code(msg["unsafe_sql"], language="sql")

        # Error SQL debug
        if msg["role"] == "assistant" and msg.get("error_sql"):
            with st.expander("🔍 See SQL that caused error"):
                st.code(msg["error_sql"], language="sql")

        # SQL results — rendered via history loop with unique idx keys
        if msg["role"] == "assistant" and msg.get("sql_result"):
            result_data = msg["sql_result"]
            render_visualization(
                result_data["df"],
                result_data["sql"],
                idx,
                result_data.get("insights", ""),
            )


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
        # ── Phase 1: Live streaming inside a single assistant bubble ───────
        # We open ONE st.chat_message block and stream everything into it.
        # Thinking goes into an open expander; SQL into a code placeholder.
        # After streaming we CLEAR the temporary widgets and render the final
        # result in the same block — no st.rerun(), no 10-second wait.

        with st.chat_message("assistant"):
            # Live thinking expander (expanded so user sees it stream)
            thinking_expander = st.expander("🧠 AI is thinking…", expanded=True)
            thinking_area = thinking_expander.empty()

            # Temporary SQL streaming area (cleared after streaming)
            sql_stream_label = st.empty()
            sql_stream_area  = st.empty()

            # Placeholder for the final summary line
            summary_area = st.empty()

            response_buffer = [""]

            def stream_handler(token: str):
                response_buffer[0] += token
                streamed = response_buffer[0]

                if "```sql" in streamed:
                    parts = streamed.split("```sql")
                    # Strip thinking markers before displaying
                    raw_thinking = (
                        parts[0]
                        .replace("💭 **Thinking:**", "")
                        .replace("Thinking:", "")
                        .strip()
                    )
                    thinking_area.markdown(raw_thinking)

                    # Stream SQL tokens live
                    sql_streamed = parts[1].split("```")[0] if len(parts) > 1 else ""
                    if sql_streamed.strip():
                        sql_stream_label.markdown("**🔄 Generating SQL…**")
                        sql_stream_area.code(sql_streamed.strip(), language="sql")
                else:
                    raw_thinking = (
                        streamed
                        .replace("💭 **Thinking:**", "")
                        .replace("Thinking:", "")
                        .strip()
                    )
                    thinking_area.markdown(raw_thinking)

            # ── Generate SQL (fully streamed) ──────────────────────────────
            result  = generate_sql(user_input, stream_callback=stream_handler)
            thinking = result["thinking"]
            sql      = result["sql"]

            # Clear the temporary SQL streaming widgets immediately
            sql_stream_label.empty()
            sql_stream_area.empty()

            # ── Phase 2: Render final result inline (no st.rerun) ──────────
            if not sql:
                answer = "⚠️ Could not generate a valid SQL query. Please rephrase your question."
                summary_area.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "thinking": thinking,
                })

            elif not is_safe_sql(sql):
                sql_lower = sql.lower().strip()
                if not (sql_lower.startswith("select") or sql_lower.startswith("with")):
                    reason = f"Query must start with SELECT or WITH. Found: {sql[:20]}..."
                else:
                    reason = "Query contains forbidden operations (INSERT/UPDATE/DELETE/DROP/etc.)"
                answer = f"⚠️ I can only run SELECT queries. {reason}"
                summary_area.markdown(answer)
                with st.expander("🔍 See generated SQL"):
                    st.code(sql, language="sql")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "thinking": thinking,
                    "unsafe_sql": sql,
                })

            else:
                try:
                    rows, cols = run_sql(sql)
                    df_res = pd.DataFrame(rows, columns=list(cols))

                    from src.text_to_sql.sql_chain import generate_insights_from_results
                    insights = generate_insights_from_results(df_res, user_input)

                    summary = f"✅ Query executed. **{len(df_res):,} rows** returned."
                    summary_area.markdown(summary)

                    # Render table + chart inline — unique key = message count before append
                    msg_key = len(st.session_state.messages)
                    render_visualization(df_res, sql, msg_key, insights)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": summary,
                        "sql_result": {"df": df_res, "sql": sql, "insights": insights},
                        "thinking": thinking,
                    })

                except Exception as e:
                    answer = f"❌ SQL Execution Error: {str(e)}"
                    summary_area.error(answer)
                    with st.expander("🔍 See SQL that caused error"):
                        st.code(sql, language="sql")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "thinking": thinking,
                        "error_sql": sql,
                    })

    # ── RAG ────────────────────────────────────────────────────────────────
    else:
        retriever = load_retriever(collection_name)
        raw_docs: list[Document] = []
        try:
            raw_docs = retriever.invoke(user_input)
        except Exception as e:
            st.warning(f"Retrieval failed: {e}")

        # BUG 1 FIX: Chunks debug panel (was removed, now restored)
        with st.expander("🔍 Retrieved chunks (highlighted)"):
            st.write(f"Retrieved **{len(raw_docs)} chunks** from the knowledge base.")
            for i, doc in enumerate(raw_docs[:6]):  # Show up to 6 chunks
                st.markdown(f"**Chunk {i+1}:**")
                st.markdown(
                    highlight_text(doc.page_content[:800], user_input),
                    unsafe_allow_html=True,
                )
                st.caption(f"Source: {doc.metadata.get('source', 'unknown')}")
                if i < len(raw_docs) - 1:
                    st.markdown("---")

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