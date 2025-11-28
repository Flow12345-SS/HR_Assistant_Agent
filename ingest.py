import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Path to HR policy PDF
pdf_path = "data/hr_policy.pdf"

# Safety check
if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
    raise ValueError("❌ hr_policy.pdf is missing or empty. Add a valid HR policy PDF in the data/ folder.")

print("📄 Loading HR policy PDF...")
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"✅ Loaded {len(documents)} pages. Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} text chunks.")

print("🧠 Creating FREE local embeddings (sentence-transformers)...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("📦 Building FAISS vector store...")
db = FAISS.from_documents(chunks, embeddings)

os.makedirs("vector_store", exist_ok=True)
db.save_local("vector_store")

print("🎉 Vector store created successfully in 'vector_store/'!")
