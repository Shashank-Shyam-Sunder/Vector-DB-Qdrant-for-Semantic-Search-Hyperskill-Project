from qdrant_client import QdrantClient
from openai import OpenAI
import os
from dotenv import load_dotenv
import re
from colorama import init, Fore, Style
init()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_LITELLM_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print(f'{Fore.RED} ERROR: OpenAI API key not found in environment variable OPENAI_LITELLM_API_KEY')
# %% STEP 1: Initiating the OPENAI client and getting embedding vector for a query
openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://litellm.aks-hs-prod.int.hyperskill.org/"
)

#  For debugging to get the list of models available
# model_list = client.models.list()
# model_names = [model.id for model in model_list]
# print(model_names)
# %%

def get_embedding(query: str) -> list:
    # query = re.sub(r'\s+', ' ', query.strip())
    query = query.strip().replace("\n", " ")
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    embedding = response.data[0].embedding
    return embedding

# For debugging
# test_embedding = get_embedding("What is the meaning of life?")
# print(test_embedding[:5], len(test_embedding))

# %% STEP 2: Initiating the Qdrant client and performing semantic search
qdrant_client = QdrantClient(host="localhost", port=6333)

def top_5_similar_papers(query: str) -> list:
    embedding = get_embedding(query)
    result = qdrant_client.query_points(
        collection_name="arxiv_papers",
        query=embedding,
        limit=5,
        with_payload=True,
        with_vectors=False
    ).points

    paper_ids = [point.payload.get("id") for point in result]

    return paper_ids

config = qdrant_client.get_collection(collection_name="arxiv_papers").config
print(config)

# Insert the name = main statement here
    # input_query = "the attention mechanism in deep learning"
    # query_result = top_5_similar_papers(input_query)
    # print(query_result)