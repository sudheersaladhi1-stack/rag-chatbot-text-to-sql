import streamlit as st
from uuid import uuid4
import os
import re
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# WORKING RAG chain + memory store
from src.rag_chat_memory import rag_chain_with_memory, store


# =====================================================
# Utilities
# =====================================================
def highlight_text(text: str, query: str):
    if not query:
        return text

    keywords = {
        w.lower()
        for w in re.findall(r"\w+", query)
        if len(w) > 2
    }

    def repl(match):
        w = match.group(0)
        return f"<mark>{w}</mark>" if w.lower() in keywords else w

    return re.sub(r"\w+", repl, text)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def extract_person_names(text: str):
    return {w.lower() for w in re.findall(r"[A-Z][a-z]+", text)}


# =====================================================
# URL Loader
# =====================================================
def load_url_as_documents(url: str):
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = "\n".join(
        line.strip()
        for line in soup.get_text("\n").splitlines()
        if line.strip()
    )

    return [{
        "page_content": text,
        "metadata": {
            "source": urlparse(url).netloc,
            "type": "url",
            "url": url
        }
    }]


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


def get_vectorstore(collection_name: str):
    return Chroma(
        collection_name=collection_name,
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )


@st.cache_resource(show_spinner=False)
def load_retriever(collection_name: str):
    return get_vectorstore(collection_name).as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )


# =====================================================
# Sidebar
# =====================================================
st.sidebar.header("🗂️ Collection")
collection_name = st.sidebar.text_input("Collection name", value="default")

st.sidebar.header("📂 Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "PDF / TXT files", type=["pdf", "txt"], accept_multiple_files=True
)

st.sidebar.header("🌐 Add Website URL")
url_input = st.sidebar.text_input("Enter website URL")

retriever = load_retriever(collection_name)


# =====================================================
# Ingest Files
# =====================================================
def ingest_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    def hash_doc(text, source):
        return hashlib.md5(
            f"{collection_name}:{source}:{text}".encode()
        ).hexdigest()

    unique = {}
    for c in chunks:
        source = c.metadata.get("source", "")
        c.metadata["collection"] = collection_name
        h = hash_doc(c.page_content, source)
        if h not in unique:
            unique[h] = c

    vs = get_vectorstore(collection_name)
    vs.add_documents(
        documents=list(unique.values()),
        ids=list(unique.keys())
    )


if st.sidebar.button("📥 Ingest documents"):
    if not uploaded_files:
        st.sidebar.warning("Upload at least one file")
    else:
        with st.spinner("Ingesting files..."):
            docs = []
            for f in uploaded_files:
                tmp = f"temp_{f.name}"
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
        with st.spinner("Fetching website..."):
            url_docs = load_url_as_documents(url_input)
            ingest_documents([
                type("Doc", (), d) for d in url_docs
            ])

        st.cache_resource.clear()
        st.sidebar.success("Website added ✅")
        st.rerun()


# =====================================================
# Clear Knowledge Base
# =====================================================
st.sidebar.divider()

if st.sidebar.button("🗑️ Clear knowledge base"):
    vs = get_vectorstore(collection_name)
    ids = vs._collection.get().get("ids", [])
    if ids:
        vs._collection.delete(ids=ids)

    store.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    st.sidebar.success("Knowledge base cleared ✅")
    st.rerun()


# =====================================================
# Session state
# =====================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================================
# Disable chat if empty
# =====================================================
doc_count = get_vectorstore(collection_name)._collection.count()
st.sidebar.caption(f"📄 Documents in DB: {doc_count}")

if doc_count == 0:
    st.info("Upload documents or a URL to start.")
    st.stop()


# =====================================================
# Display history
# =====================================================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# =====================================================
# Chat
# =====================================================
user_input = st.chat_input("Ask a question based on the uploaded knowledge")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    raw_docs = retriever.invoke(user_input)

    seen, retrieved = set(), []
    for d in raw_docs:
        t = d.page_content.strip()
        if t and t not in seen:
            seen.add(t)
            retrieved.append(d)
        if len(retrieved) == 3:
            break

    if not retrieved:
        st.chat_message("assistant").markdown(
            "I don't know based on the provided context."
        )
        st.stop()

    context_text = format_docs(retrieved)

    # 🚫 PERSON MISMATCH BLOCK
    if extract_person_names(user_input) - extract_person_names(context_text):
        st.chat_message("assistant").markdown(
            "I don't know based on the provided context."
        )
        st.stop()

    with st.chat_message("assistant"):
        response = rag_chain_with_memory.invoke(
            {"input": user_input, "context": context_text},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        st.markdown(response)

        with st.expander("🔍 Show retrieved context"):
            for i, d in enumerate(retrieved, 1):
                st.markdown(
                    f"**Chunk {i}**\n\n{highlight_text(d.page_content, user_input)}",
                    unsafe_allow_html=True
                )

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "docs": retrieved,
        "query": user_input
    })
