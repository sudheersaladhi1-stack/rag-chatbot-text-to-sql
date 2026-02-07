from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI


# Load embeddings
embedding_model = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# Load vector DB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# System prompt
system_prompt = """You are a retrieval-augmented assistant.

You MUST answer the question using ONLY the provided context.
Do NOT use prior knowledge.
Do NOT guess.
Do NOT explain concepts that are not explicitly present in the context.

If the answer cannot be found in the context, reply exactly with:
"I don't know based on the provided context."

Context:
{context}
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# Format documents
def format_docs(docs):
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs)

# Local LLM
# llm = Ollama(
#     model="mistral",
#     temperature=0
# )
llm = ChatOpenAI(
    model="gpt-4o-mini",   # best cost/quality for RAG
    temperature=0.2,
)

# Build LCEL RAG chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Test
if __name__ == "__main__":
    question = "What is machine learning?"
    answer = rag_chain.invoke(question)
    print("\nQuestion:", question)
    print("Answer:", answer)
