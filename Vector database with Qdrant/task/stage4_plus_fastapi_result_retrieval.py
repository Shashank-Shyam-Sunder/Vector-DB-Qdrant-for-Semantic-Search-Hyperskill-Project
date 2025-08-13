from fastapi import FastAPI
from typing import Optional, List, Dict
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, ScoredPoint, MatchText
from openai import OpenAI
import os
from dotenv import load_dotenv
import re
from colorama import init, Fore, Style
from typing import Optional
init()
load_dotenv()

# Creating an instance of FastAPI Application
app = FastAPI()

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

def get_author_name(query:str) -> Optional[str]:
    search_pattern =  r"by\s+([A-Za-z\s\-]+)"
    match = re.search(search_pattern, query)
    if match:
        return match.group(1)
    else:
        return None


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

def top_k_similar_papers(query: str, k: int = 5) -> list:
    embedding = get_embedding(query)
    author_name = get_author_name(query)
    if author_name:
        # Filter by author name
        match_text_filter = Filter(
            must=[FieldCondition(key="authors", match=MatchText(text=author_name))]
        )
        result = qdrant_client.query_points(
            collection_name="arxiv_papers",
            query=embedding,
            limit=k,
            with_payload=True,
            with_vectors=False,
            query_filter=match_text_filter
        ).points
    else:
        result = qdrant_client.query_points(
            collection_name="arxiv_papers",
            query=embedding,
            limit=3,
            with_payload=True,
            with_vectors=False
        ).points

    # paper_ids = [point.payload.get("id") for point in result]
    return result

# %% STEP 3: Defining the FastAPI endpoint for semantic search
class SearchRequest(BaseModel):
    """
    Model representing a search request sent by a client.

    Attributes:
        query (str): The search query string provided by the client.
        top_n (Optional[int]): The number of top search results to return.
            Defaults to 5 if not provided.
    """
    query: str
    top_n: Optional[int] = 5

class SearchResult(BaseModel):
    """
    Model representing an individual search result.

    Attributes:
        id (str): Unique identifier for the search result item.
        payload (Dict): A dictionary containing additional data or metadata about the item.
        score (float): The relevance score of the search result, typically used for ranking.
    """
    id: str
    payload: Dict
    score: float

class SearchResponse(BaseModel):
    """
    Model representing the response returned to a client after a search.

    Attributes:
        results (List[SearchResult]): A list containing the top matching search results.
    """
    results: List[SearchResult]

@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    query = request.query
    top_n = request.top_n
    search_results = top_k_similar_papers(query, top_n)
    results_list = []
    for result in search_results:
        results_list.append(SearchResult(id=result.payload.get("id"), payload=result.payload, score=result.score))

    return SearchResponse(results=results_list)

