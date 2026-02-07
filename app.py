import streamlit as st
from uuid import uuid4
import os
import re
import hashlib

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Import your RAG chain (Step 6)
from src.rag_chat_memory import rag_chain_with_memory


# =====================================================
# Utility: Highlight matched text (STEP 10)
# =====================================================
def highlight_text(text: str, query: str):
    """
    Highlight query keywords inside retrieved text
    """
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


# =====================================================
# Streamlit config
# =====================================================
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chatbot")
st.caption("Chat + File Upload + Retrieved Context Highlighting")


# =====================================================
# Embeddings & Vectorstore helpers
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


@st.cache_resource
def load_retriever(collection_name: str):
    vectorstore = get_vectorstore(collection_name)
    return vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3
    }
    )

# =====================================================
# Sidebar: File upload & incremental ingestion (STEP 9)
# =====================================================
st.sidebar.header("🗂️ Select Collection")

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

                if file.name.endswith(".pdf"):
                    loader = PyPDFLoader(temp_path)
                else:
                    loader = TextLoader(temp_path, encoding="utf-8")

                all_docs.extend(loader.load())
                os.remove(temp_path)

            splitter = RecursiveCharacterTextSplitter(
                 chunk_size=600,
                chunk_overlap=150,
                separators=["\n\n", "\n", ".", " "]
                    )

            chunks = splitter.split_documents(all_docs)
            def doc_hash(text: str) -> str:
             return hashlib.md5(text.encode("utf-8")).hexdigest()
            
            unique_chunks = {}
            for chunk in chunks:
                 h = doc_hash(chunk.page_content)
                 if h not in unique_chunks:
                        unique_chunks[h] = chunk
            vectorstore = get_vectorstore(collection_name)
            vectorstore.add_documents(chunks)

        st.sidebar.success("Documents added successfully ✅")


# =====================================================
# Session state
# =====================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================================
# Display chat history (with highlighted context)
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
                        f"""
                        **Chunk {i}**

                        {highlighted}
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown("---")


# =====================================================
# Chat input
# =====================================================
user_input = st.chat_input(
    "Ask a question based on the uploaded documents..."
)

if user_input:
    # --- User message ---
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- Retrieve context (for display only) ---
    retrieved_docs = retriever.invoke(user_input)
    seen = set()
    filtered_docs = []

    for doc in retrieved_docs:
        key = doc.page_content.strip()
        if key not in seen:
            seen.add(key)
            filtered_docs.append(doc)

    retrieved_docs = filtered_docs


    # --- Generate answer ---
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain_with_memory.invoke(
                {"input": user_input},
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
                        f"""
                        **Chunk {i}**

                        {highlighted}
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown("---")

    # --- Save assistant message ONCE ---
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "docs": retrieved_docs,
            "query": user_input
        }
    )
