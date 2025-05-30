# src/main.py

from fastapi import FastAPI
from dotenv import load_dotenv
import os

# 1. Load env vars (so os.getenv can pick them up)
load_dotenv()

# 2. Create the FastAPI app
app = FastAPI()

# 3. Your existing root endpoint
@app.get("/")
async def root():
    return {"status": "ok"}

# --- New code below ---

# 4. Import your chat router
from routes.chat import router as chat_router

# 5. Mount it under the default path (/chat)
app.include_router(chat_router)
