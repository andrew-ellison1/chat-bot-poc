# src/main.py

from fastapi import FastAPI
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI()

# -------------------------------------------------------
# 1. Initialize Pinecone client
from pinecone import Pinecone

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env     = os.getenv("PINECONE_ENV")
index_name       = "resume-index"

pc = Pinecone(
    api_key=pinecone_api_key,
    environment=pinecone_env
)
# Get a handle to the index (assumes it already exists)
pinecone_index = pc.Index(index_name)
# -------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok"}

# We’ll include the /chat router farther down...
from routes.chat import router as chat_router
app.include_router(chat_router)
