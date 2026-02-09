import streamlit as st
from uuid import uuid4
import os
import re
import hashlib

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Import your RAG chain (WORKING ONE)
from src.rag_chat_memory import rag_chain_with_memory


# =====================================================
# Utility: Highlight matched text
# =====================================================
def highlight_text(text: str, query: str):
    if not query:
        return text

    keywords = {
        word.lower()
        for word in re.findall(r"\w+", query)
        if len(word) > 2
    }

    def replacer(match):
        word = match.group(0)
        if word.lower() in keywords:
            return f"<mark>{word}</mark>"
        return word

    return re.sub(r"\w+", replacer, text)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# =====================================================
# Streamlit config
# =====================================================
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chatbot")
st.caption("Chat + File Upload + Retrieved Context (3 chunks guaranteed)")


# =====================================================
# Embeddings & Vectorstore
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
    vectorstore = get_vectorstore(collection_name)
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )


def clear_collection(collection_name: str):
    try:
        vectorstore = get_vectorstore(collection_name)
        client = vectorstore._client

        # Delete collection safely
        client.delete_collection(collection_name)

        # Clear Streamlit cache so retriever reloads cleanly
        st.cache_resource.clear()

        return True
    except Exception as e:
        st.error(f"Failed to clear DB: {e}")
        return False


# =====================================================
# Sidebar: Collection + Upload
# =====================================================
st.sidebar.header("🗂️ Collection")

collection_name = st.sidebar.text_input(
    "Collection name",
    value="default",
    help="Each collection is an independent knowledge base"
)

st.sidebar.header("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

retriever = load_retriever(collection_name)

# -----------------------------------------------------
# Ingest documents
# -----------------------------------------------------
if st.sidebar.button("📥 Ingest documents"):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one file.")
    else:
        with st.spinner("Ingesting documents..."):
            all_docs = []

            for file in uploaded_files:
                temp_path = f"temp_{file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file.read())

                loader = (
                    PyPDFLoader(temp_path)
                    if file.name.endswith(".pdf")
                    else TextLoader(temp_path, encoding="utf-8")
                )

                all_docs.extend(loader.load())
                os.remove(temp_path)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=150,
                separators=["\n\n", "\n", ".", " "]
            )

            chunks = splitter.split_documents(all_docs)

            # 🔐 Strong deduplication (collection + source aware)
            def doc_hash(text: str, source: str) -> str:
                return hashlib.md5(
                    f"{collection_name}:{source}:{text}".encode("utf-8")
                ).hexdigest()

            unique_chunks = {}
            for chunk in chunks:
                chunk.metadata["collection"] = collection_name
                source = chunk.metadata.get("source", "")
                h = doc_hash(chunk.page_content, source)
                if h not in unique_chunks:
                    unique_chunks[h] = chunk

            vectorstore = get_vectorstore(collection_name)

            documents = []
            ids = []

            for h, chunk in unique_chunks.items():
                documents.append(chunk)
                ids.append(h)

            vectorstore.add_documents(
                documents=documents,
                ids=ids
            )

        st.sidebar.success("Documents added successfully ✅")
        st.cache_resource.clear()
        st.rerun()
# -----------------------------------------------------
# CLEAR DATABASE BUTTON (🔥 IMPORTANT)
# -----------------------------------------------------
st.sidebar.divider()

if st.sidebar.button("🗑️ Clear knowledge base"):
    if clear_collection(collection_name):
        st.session_state.messages = []
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
# Disable chat if DB empty
# =====================================================
vectorstore = get_vectorstore(collection_name)
doc_count = vectorstore._collection.count()

st.sidebar.caption(f"📄 Documents in DB: {doc_count}")

if doc_count == 0:
    st.info("📂 Knowledge base is empty. Upload documents to start.")
    st.stop()


# =====================================================
# Display chat history
# =====================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "docs" in msg:
            with st.expander("🔍 Show retrieved context"):
                for i, doc in enumerate(msg["docs"], start=1):
                    highlighted = highlight_text(
                        doc.page_content,
                        msg.get("query", "")
                    )
                    st.markdown(
                        f"**Chunk {i}**\n\n{highlighted}",
                        unsafe_allow_html=True
                    )
                    st.markdown("---")


# =====================================================
# Chat input
# =====================================================
user_input = st.chat_input("Ask a question based on the uploaded documents...")

if user_input:
    # User message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # -------------------------------------------------
    # Retrieve EXACTLY 3 UNIQUE chunks
    # -------------------------------------------------
    raw_docs = retriever.invoke(user_input)

    seen = set()
    retrieved_docs = []

    for doc in raw_docs:
        text = doc.page_content.strip()
        if text and text not in seen:
            seen.add(text)
            retrieved_docs.append(doc)
        if len(retrieved_docs) == 3:
            break

    context_text = format_docs(retrieved_docs)

    # -------------------------------------------------
    # Generate answer
    # -------------------------------------------------
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain_with_memory.invoke(
                {
                    "input": user_input,
                    "context": context_text
                },
                config={
                    "configurable": {
                        "session_id": st.session_state.session_id
                    }
                }
            )

            st.markdown(response)

            with st.expander("🔍 Show retrieved context"):
                for i, doc in enumerate(retrieved_docs, start=1):
                    highlighted = highlight_text(
                        doc.page_content,
                        user_input
                    )
                    st.markdown(
                        f"**Chunk {i}**\n\n{highlighted}",
                        unsafe_allow_html=True
                    )
                    st.markdown("---")

    # Save assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "docs": retrieved_docs,
            "query": user_input
        }
    )
