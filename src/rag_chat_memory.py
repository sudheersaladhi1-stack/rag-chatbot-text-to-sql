from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory

import os
from dotenv import load_dotenv

load_dotenv()

# =====================================================
# LLM (OpenAI)
# =====================================================
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# =====================================================
# STRICT QA PROMPT (NO HISTORY, NO ASSUMPTIONS)
# =====================================================
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You MUST answer strictly and only from the provided Context.

Rules:
- Use ONLY information explicitly present in Context.
- DO NOT use prior chat history, world knowledge, or assumptions.
- DO NOT infer or guess missing information.
- If the answer is NOT explicitly stated in Context, reply EXACTLY:
"I don't know based on the provided context."

Context:
{context}
"""
        ),
        ("human", "{input}")
    ]
)

# =====================================================
# RAG CHAIN (NO RETRIEVAL INSIDE)
# =====================================================
rag_chain = (
    qa_prompt
    | llm
    | StrOutputParser()
)

# =====================================================
# CHAT MEMORY (SAFE – DOES NOT AFFECT FACTS)
# =====================================================
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

rag_chain_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
