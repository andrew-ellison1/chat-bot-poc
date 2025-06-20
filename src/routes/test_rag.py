# test_rag.py
import asyncio, os
from dotenv import load_dotenv
from src.utils.rag import fetch_context

load_dotenv()

async def main():
    result = await fetch_context("Tell me about Andrew's experience", top_k=3)
    print("CONTEXT:\n", result)

asyncio.run(main())
