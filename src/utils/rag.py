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


async def fetch_context(question: str, top_k: int = 5) -> str:
    # — Embed the question —
    emb_resp = openai.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    )
    try:
        q_emb = emb_resp["data"][0]["embedding"]
    except Exception:
        q_emb = emb_resp.data[0].embedding

    # — Query Pinecone for more than we’ll actually need —
    query_resp = pinecone_index.query(
        vector=q_emb,
        top_k=10,                  # fetch a few extra
        include_metadata=True
    )

    # — Pull out the raw matches list (object or dict style) —
    matches = getattr(query_resp, "matches", query_resp.get("matches", []))

    # — Sort by your start_date metadata (newest first) —
    #   defaulting any missing date to “1970-01-01”
    def _get_date(m):
        md = m.metadata if hasattr(m, "metadata") else m.get("metadata", {})
        return md.get("start_date", "1970-01-01")

    sorted_matches = sorted(matches, key=_get_date, reverse=True)

    # — Now take only the top_k most recent ones —
    selected = sorted_matches[:top_k]

    # — Extract the “text” from each match into snippets —
    snippets = []
    for m in selected:
        md = m.metadata if hasattr(m, "metadata") else m.get("metadata", {})
        snippets.append(md.get("text", ""))

    # — Debug —
    print(f"🔍 fetch_context returning {len(snippets)} snippets (sorted by date)”")

    if snippets:
        return "\n\n".join(snippets)
    else:
        return "“I’m sorry, I don’t have that detail in my resume excerpts.”"
