# src/routes/chat.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import openai
import os

# If you’re using RAG, import your helper; otherwise omit this line:
# from utils.rag import fetch_context

router = APIRouter()
openai.api_key = os.getenv("OPENAI_API_KEY")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # 1. Get the user’s question
    user_q = req.message

    # 2. (Optional) fetch context with RAG
    # context = await fetch_context(user_q)
    # prompt = (
    #     "You are Andrew’s interactive resume assistant.\n\n"
    #     f"{context}\n\n"
    #     "If the answer is not here, say “I’m sorry, I don’t have that detail.”"
    # )
    # messages = [
    #     {"role": "system", "content": prompt},
    #     {"role": "user", "content": user_q},
    # ]

    # 2b. If you’re *not* using RAG yet, just send the user message:
    messages = [
        {"role": "user", "content": user_q}
    ]

    try:
        # 3. Call the new v1+ Chat API
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        # 4. Extract the assistant’s reply
        # The v1+ SDK returns a dict-like; we index into it:
        reply = resp.choices[0].message.content

        return {"reply": reply}

    except Exception as e:
        # 5. If anything goes wrong, return HTTP 500 with the error text
        raise HTTPException(status_code=500, detail=str(e))
