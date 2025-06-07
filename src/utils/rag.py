import os
import openai
from pinecone import Pinecone

#load .env
from dotenv import load_dotenv
load_dotenv()

#Reinitialize Pinecone
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV"),
)
pinecone_index = pc.Index(os.getenv("INDEX_NAME","resume-index"))

async def fetch_context(question: str, top_k: int = 3) -> str:

        
    # Embed the users questions

    resp = openai.embeddings.create(
            input=question,
            model="text-embedding-3-small"
    )