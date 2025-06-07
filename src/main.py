# src/main.py

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pinecone import Pinecone

load_dotenv()
app = FastAPI()

# 1. Mount static assets under /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Serve index.html at the root path
@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")

# 3. Health check (optional)
@app.get("/health")
async def health():
    return {"status": "ok"}

# 4. Initialize Pinecone
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)
pinecone_index = pc.Index("resume-index")

# 5. Include your chat router
from routes.chat import router as chat_router
app.include_router(chat_router)
