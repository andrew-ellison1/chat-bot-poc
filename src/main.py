# src/main.py

import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Load .env
load_dotenv()

app = FastAPI()

# Base dir is the project root (one level above src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# 1. Serve static assets from the project root’s static/ folder
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)

# 2. Serve index.html at root
@app.get("/")
async def serve_ui():
    return FileResponse(BASE_DIR / "static" / "index.html")

# 3. Health check
@app.get("/health")
async def health():
    return {"status": "ok"}

# 4. Initialize Pinecone (unchanged)
from pinecone import Pinecone

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)
pinecone_index = pc.Index(os.getenv("INDEX_NAME", "resume-index"))

# 5. Include the chat router from src.routes.chat
from src.routes.chat import router as chat_router
app.include_router(chat_router)
