# src/utils/rag.py

import os
import openai
from dotenv import load_dotenv
from pinecone import Pinecone

# 1. Load .env and set API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 2. Init Pinecone client and index handle
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)
pinecone_index = pc.Index("resume-index")


async def fetch_context(question: str, top_k: int = 3) -> str:
    """
    1) Embed the incoming question.
    2) Query Pinecone for the top_k matches.
    3) Return the joined text snippets, or an explicit fallback if none found.
    """
    # — Embed the question —
    emb_resp = openai.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    )
    # If emb_resp is a dict (v0.28+) or object, pull it out accordingly:
    try:
        q_emb = emb_resp["data"][0]["embedding"]
    except Exception:
        q_emb = emb_resp.data[0].embedding

    # — Query Pinecone —
    query_resp = pinecone_index.query(
        vector=q_emb,
        top_k=top_k,
        include_metadata=True
    )

    # — Extract the snippets —
    snippets = []
    # Try both dict- and object-style APIs:
    matches = getattr(query_resp, "matches", query_resp.get("matches", []))
    for m in matches:
        # object-like
        if hasattr(m, "metadata") and isinstance(m.metadata, dict):
            snippets.append(m.metadata.get("text", ""))
        # dict-like
        elif isinstance(m, dict) and "metadata" in m:
            snippets.append(m["metadata"].get("text", ""))

    # — Debug: print what we found —
    print(f"🔍 fetch_context received {len(snippets)} snippets for question: “{question}”")
    for i, s in enumerate(snippets, 1):
        print(f"  {i}. {s[:80]}{'…' if len(s)>80 else ''}")

    # — Return joined text or a clear fallback —
    if snippets:
        return "\n\n".join(snippets)
    else:
        return "“I’m sorry, I don’t have that detail in my resume excerpts.”"
