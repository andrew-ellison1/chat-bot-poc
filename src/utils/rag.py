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

async def query_pinecone(query: str, top_k: int = 3) -> str:
        """
    - Embeds `question` using OpenAI embeddings.
    - Queries `resume-index` for the top_k most similar chunks.
    - Returns the concatenated chunk texts (joined by line breaks).
    """
        
    # Embed the users questions

    emb_resp = openai.embeddings.create(
            input=question
    )