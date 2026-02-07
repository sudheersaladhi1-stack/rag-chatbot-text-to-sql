from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import shutil


# Load document (path relative to project root)
loader = TextLoader("data/doc_0.txt", encoding="utf-8")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")
print(chunks[0].page_content)

# LangChain-compatible embeddings
embedding_model = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Reset vector DB (dev only)
shutil.rmtree("chroma_db", ignore_errors=True)

# Store in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

# Test retrieval
results = vectorstore.similarity_search(
    "What is machine learning?", k=2
)

for doc in results:
    print(doc.page_content)
