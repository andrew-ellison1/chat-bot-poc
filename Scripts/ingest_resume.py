# scripts/ingest_resume.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai, os
from pinecone import Pinecone
from dotenv import load_dotenv

# 1) Load env
load_dotenv()

# 2) Init Pinecone & OpenAI
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
index = pc.Index(os.getenv("INDEX_NAME", "resume-index"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# 3) Load & split

loader = PyPDFLoader("static/resume.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100,
        separators=["\n\n", "\n", "•", " ", ""])
chunks = splitter.split_documents(docs)

# 4) Embed & upsert
vectors = []
for i, doc in enumerate(chunks):
    resp = openai.embeddings.create(input=doc.page_content, model="text-embedding-ada-002")
    emb = resp.data[0].embedding
    vectors.append((f"chunk_{i}", emb, {"text": doc.page_content}))
index.upsert(vectors)
print(f"✅ Upserted {len(vectors)} chunks")
