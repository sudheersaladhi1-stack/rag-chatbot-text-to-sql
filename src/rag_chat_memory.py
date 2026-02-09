from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# =====================================================
# LLM
# =====================================================
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# =====================================================
# STRICT QA PROMPT (context-only)
# =====================================================
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a retrieval-augmented assistant.

Answer the user's question ONLY using the provided context.

If the answer IS present in the context:
- Answer clearly and directly.

If the answer is NOT present in the context:
- Reply EXACTLY with:
"I don't know based on the provided context."

Context:
{context}
"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

# =====================================================
# RAG chain (NO RETRIEVER HERE)
# =====================================================
rag_chain = (
    RunnablePassthrough
    | qa_prompt
    | llm
    | StrOutputParser()
)

# =====================================================
# Memory store
# =====================================================
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# =====================================================
# Chain with memory
# =====================================================
rag_chain_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
