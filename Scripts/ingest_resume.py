# scripts/ingest_resume.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai, os, re
from datetime import datetime
from pinecone import Pinecone
from dotenv import load_dotenv

# 1) Load environment variables
load_dotenv()

# 2) Initialize Pinecone & OpenAI
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)
index = pc.Index(os.getenv("INDEX_NAME", "resume-index"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# Helper to extract the first "Month Year" pattern for start_date
MONTH_YEAR_REGEX = re.compile(r"([A-Za-z]+ \d{4})")
def extract_start_date(text: str) -> str:
    match = MONTH_YEAR_REGEX.search(text)
    if match:
        try:
            # parse e.g. "January 2021" → ISO "2021-01-01"
            dt = datetime.strptime(match.group(1), "%B %Y").date()
            return dt.isoformat()
        except ValueError:
            pass
    # fallback to epoch for unknown dates
    return "1970-01-01"

# 3) Load & split resume PDF
loader = PyPDFLoader(os.path.join(os.getcwd(), "static", "resume.pdf"))
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", "•", " ", ""]
)
chunks = splitter.split_documents(docs)

# 4) Embed & upsert with date metadata
vectors = []
for i, doc in enumerate(chunks):
    # embed the chunk
    resp = openai.embeddings.create(
        input=doc.page_content,
        model="text-embedding-ada-002"
    )
    emb = resp.data[0].embedding
    # extract start_date metadata
    start_date = extract_start_date(doc.page_content)
    metadata = {
        "text": doc.page_content,
        "start_date": start_date
    }
    vectors.append((f"chunk_{i}", emb, metadata))

# upsert all vectors into Pinecone
index.upsert(vectors)
print(f"✅ Upserted {len(vectors)} chunks with start_date metadata")
