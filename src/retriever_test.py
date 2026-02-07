from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Initialize the SAME embedding model used during ingestion
embedding_model = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load persisted Chroma DB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Test query
query = "What is machine learning?"
docs = retriever.invoke(query)

# Inspect results
print(f"\nQuery: {query}")
for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
