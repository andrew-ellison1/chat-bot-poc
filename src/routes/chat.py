from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import openai, os
from utils.rag import fetch_context   # ← must import your helper

router = APIRouter()
openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    user_q = req.message

    # 1. Fetch the top-3 relevant chunks for this question
    context_text = await fetch_context(user_q, top_k=3)

    # DEBUG (temporary): log what you retrieved
    print("🔍 Retrieved context:\n", context_text)

    # 2. Build your messages array—include the context in the system role
    messages = [
        {
            "role": "system",
            "content": (
                "You are Andrew’s interactive resume assistant.\n\n"
                f"{context_text}\n\n"
                "If the answer is not in these excerpts, say “I’m sorry, I don’t have that detail.”"
            )
        },
        {"role": "user", "content": user_q}
    ]

    try:
        # 3. Call the new v1+ Chat API
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        # 4. Extract the reply
        reply = resp.choices[0].message.content
        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
