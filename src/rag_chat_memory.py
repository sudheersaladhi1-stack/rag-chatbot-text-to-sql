from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# Embeddings + Vector DB
# -------------------------
embedding_model = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------
# LLM (OpenAI)
# -------------------------
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# -------------------------
# Rewrite follow-up questions
# -------------------------
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's question as a standalone question using the chat history."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

rewrite_chain = rewrite_prompt | llm | StrOutputParser()

def retrieve_with_history(inputs: dict):
    standalone = rewrite_chain.invoke(inputs)
    return retriever.invoke(standalone)

history_aware_retriever = RunnableLambda(retrieve_with_history)

# -------------------------
# Final QA prompt
# -------------------------
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a retrieval-augmented assistant.

Answer the user's question using ONLY the provided context.

Rules:
- You MAY summarize, infer, or rephrase information that is clearly present in the context.
- If the context contains enough information to answer the question, provide a clear and concise answer.
- DO NOT use chat history, prior knowledge, or assumptions.
- If the context does NOT contain sufficient information, reply EXACTLY with:
"I don't know based on the provided context."

Context:
{context}
"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    RunnablePassthrough.assign(
        context=history_aware_retriever | format_docs
    )
    | qa_prompt
    | llm
    | StrOutputParser()
)

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