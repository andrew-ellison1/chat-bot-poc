from dotenv import load_dotenv
load_dotenv()    # reads .env into os.environ

import os
from pinecone import Pinecone

# now these will be populated
api_key = os.getenv("PINECONE_API_KEY")
env    = os.getenv("PINECONE_ENV")

pc = Pinecone(api_key=api_key, environment=env)
print("Indexes:", pc.list_indexes().names())
