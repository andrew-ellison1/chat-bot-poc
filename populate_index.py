import os
from dotenv import load_dotenv
import openai
from pinecone import pinecone

#load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pc = pinecone(api_key=os.getenv("PINECONE_API_KEY"),
               environment=os.getenv("PINECONE_ENV"))

#get index
index_name = "resume-index"
if index_name not in pc.list_index().names():pc.create_index(name=index_name, dimension=384, metric="cosine")
index = pc.index(index_name)

print("Index created:", index_name)