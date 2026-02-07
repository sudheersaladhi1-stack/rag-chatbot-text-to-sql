from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------
# Load embeddings and vector DB
# -------------------------------------------------
embedding_model = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

#retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,          # final chunks returned
        "fetch_k": 10,   # candidate pool
        "lambda_mult": 0.4
    }
)
# -------------------------------------------------
# LLM
# -------------------------------------------------
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]


# llm = HuggingFaceEndpoint(
#     repo_id="google/flan-t5-large",
#     task="text2text-generation",
#     temperature=0.2,
#     max_new_tokens=512,
# )

llm = ChatOpenAI(
    model="gpt-4o-mini",   # best cost/quality for RAG
    temperature=0.2,
)

# -------------------------------------------------
# Rewrite prompt (history-aware question)
# -------------------------------------------------
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Rewrite the user's question as a standalone question using the chat history."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

rewrite_chain = rewrite_prompt | llm | StrOutputParser()


# -------------------------------------------------
# History-aware retrieval (IMPORTANT FIX)
# -------------------------------------------------
def retrieve_with_history(inputs: dict):
    # inputs contains: {"input": ..., "chat_history": [...]}
    standalone_question = rewrite_chain.invoke(inputs)
    docs = retriever.invoke(standalone_question)
    return docs

history_aware_retriever = RunnableLambda(retrieve_with_history)


# -------------------------------------------------
# Final QA prompt
# -------------------------------------------------
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a retrieval-augmented assistant.

Answer the user's question ONLY using the provided context.

If the answer IS present in the context:
- Answer directly and concisely.
- Do NOT say "I don't know".

If the answer is NOT present in the context:
- Reply EXACTLY with:
"I don't know based on the provided context."
- Do NOT add any extra explanation.
- Do NOT use outside knowledge.

Context:
{context}
"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)


def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs, start=1):
        formatted.append(
            f"""
[Document {i}]
{doc.page_content.strip()}
"""
        )
    return "\n".join(formatted)



# -------------------------------------------------
# LCEL RAG chain (PASS chat_history THROUGH)
# -------------------------------------------------
rag_chain = (
    RunnablePassthrough.assign(
        context=history_aware_retriever | format_docs
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)


# -------------------------------------------------
# Chat memory store
# -------------------------------------------------
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


rag_chain_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


# -------------------------------------------------
# Test
# -------------------------------------------------
if __name__ == "__main__":
    session_id = "demo"

    print("\nQ1:")
    print(
        rag_chain_with_memory.invoke(
            {"input": "What is machine learning?"},
            config={"configurable": {"session_id": session_id}},
        )
    )

    print("\nQ2 (follow-up):")
    print(
        rag_chain_with_memory.invoke(
            {"input": "Explain it simply"},
            config={"configurable": {"session_id": session_id}},
        )
    )

    print("\nQ3 (out of scope):")
    print(
        rag_chain_with_memory.invoke(
            {"input": "What is quantum physics?"},
            config={"configurable": {"session_id": session_id}},
        )
    )
