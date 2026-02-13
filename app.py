import streamlit as st
from uuid import uuid4
import os, re, hashlib, requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

# RAG chain + memory store
from src.rag_chat_memory import rag_chain_with_memory, store

import pandas as pd
from src.text_to_sql.sql_chain import generate_sql
from src.text_to_sql.sql_guard import is_safe_sql
from src.text_to_sql.db import run_sql


# =====================================================
# Utilities
# =====================================================
def highlight_text(text: str, query: str):
    if not query:
        return text
    keywords = {w.lower() for w in re.findall(r"\w+", query) if len(w) > 2}
    return re.sub(
        r"\w+",
        lambda m: f"<mark>{m.group(0)}</mark>"
        if m.group(0).lower() in keywords
        else m.group(0),
        text,
    )


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def extract_person_names(text: str):
    return {w.lower() for w in re.findall(r"[A-Z][a-z]+", text)}


# =====================================================
# URL Loader
# =====================================================
def load_url_as_documents(url: str):
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        t.decompose()

    text = "\n".join(
        line.strip()
        for line in soup.get_text("\n").splitlines()
        if line.strip()
    )

    return [
        Document(
            page_content=text,
            metadata={
                "source": urlparse(url).netloc,
                "type": "url",
                "url": url,
            },
        )
    ]


# =====================================================
# Streamlit config
# =====================================================
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chatbot")
st.caption("PDF / TXT / URL → Strict RAG (No Hallucination)")


# =====================================================
# Vectorstore
# =====================================================
embedding_model = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


def get_vectorstore(collection):
    return Chroma(
        collection_name=collection,
        persist_directory="chroma_db",
        embedding_function=embedding_model,
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

st.sidebar.header("📂 Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "PDF / TXT files", type=["pdf", "txt"], accept_multiple_files=True
)

st.sidebar.divider()
mode = st.sidebar.radio(
    "Chat Mode",
    ["📄 Document Q&A (RAG)", "📊 Database Q&A (Text-to-SQL)"]
)


st.sidebar.header("🌐 Add Website URL")
url_input = st.sidebar.text_input("Enter website URL")

retriever = load_retriever(collection_name)


# =====================================================
# Ingest helper
# =====================================================
def ingest_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    def make_id(text, src):
        return hashlib.md5(f"{collection_name}:{src}:{text}".encode()).hexdigest()

    unique = {}
    for c in chunks:
        src = c.metadata.get("source", "")
        c.metadata["collection"] = collection_name
        uid = make_id(c.page_content, src)
        unique[uid] = c

    vs = get_vectorstore(collection_name)
    vs.add_documents(list(unique.values()), ids=list(unique.keys()))


# =====================================================
# Ingest files
# =====================================================
if st.sidebar.button("📥 Ingest documents"):
    if not uploaded_files:
        st.sidebar.warning("Upload at least one file")
    else:
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


# =====================================================
# Ingest URL
# =====================================================
if st.sidebar.button("🌍 Ingest URL"):
    if not url_input:
        st.sidebar.warning("Enter a valid URL")
    else:
        ingest_documents(load_url_as_documents(url_input))
        st.cache_resource.clear()
        st.sidebar.success("Website added ✅")
        st.rerun()


# =====================================================
# Clear Knowledge Base (SAFE)
# =====================================================
st.sidebar.divider()
if st.sidebar.button("🗑️ Clear knowledge base"):
    vs = get_vectorstore(collection_name)
    ids = vs._collection.get().get("ids", [])
    if ids:
        vs._collection.delete(ids=ids)

    store.clear()
    st.cache_resource.clear()
    st.session_state["messages"] = []
    st.session_state["session_id"] = str(uuid4())

    st.sidebar.success("Knowledge base cleared ✅")
    st.rerun()


# =====================================================
# Session State (FIXED)
# =====================================================
st.session_state.setdefault("session_id", str(uuid4()))
st.session_state.setdefault("messages", [])


# =====================================================
# Disable chat if DB empty
# =====================================================
doc_count = get_vectorstore(collection_name)._collection.count()
st.sidebar.caption(f"📄 Documents in DB: {doc_count}")

if doc_count == 0:
    st.info("Upload documents or a URL to start.")
    st.stop()


# =====================================================
# Display chat history (SOURCE OF TRUTH)
# =====================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =====================================================
# Chat
# =====================================================
user_input = st.chat_input("Ask a question")

if user_input:
    # -------------------------------
    # 1️⃣ USER MESSAGE (always show)
    # -------------------------------
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # -------------------------------
    # 2️⃣ MODE SWITCH
    # -------------------------------
    if mode == "📊 Database Q&A (Text-to-SQL)":
        from src.text_to_sql.sql_chain import generate_sql
        from src.text_to_sql.sql_guard import is_safe_sql
        from src.text_to_sql.db import run_sql
        import pandas as pd

        with st.chat_message("assistant"):
            with st.spinner("Generating SQL..."):
                sql = generate_sql(user_input)

            if not is_safe_sql(sql):
                answer = "I don't know based on the provided context."
                st.markdown(answer)
            else:
                st.markdown("**Generated SQL:**")
                st.code(sql, language="sql")

                try:
                    rows, cols = run_sql(sql)
                    df = pd.DataFrame(rows, columns=cols)

                    st.markdown("**Query Result:**")
                    st.dataframe(df)

                    answer = "Here are the results based on your data."

                except Exception as e:
                    answer = f"Database error: {e}"
                    st.error(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    # -------------------------------
    # 3️⃣ RAG MODE (STRICT)
    # -------------------------------
    else:
        raw_docs = retriever.invoke(user_input)

        seen, docs = set(), []
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

            # 🚫 PERSON NAME MISMATCH GUARD
            if extract_person_names(user_input) - extract_person_names(context):
                answer = "I don't know based on the provided context."
            else:
                answer = rag_chain_with_memory.invoke(
                    {"input": user_input, "context": context},
                    config={
                        "configurable": {
                            "session_id": st.session_state.session_id
                        }
                    },
                )

        with st.chat_message("assistant"):
            st.markdown(answer)

            if docs:
                with st.expander("🔍 Show retrieved context"):
                    for i, d in enumerate(docs, 1):
                        st.markdown(
                            f"**Chunk {i}**\n\n{highlight_text(d.page_content, user_input)}",
                            unsafe_allow_html=True,
                        )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
