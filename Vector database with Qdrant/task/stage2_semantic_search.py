from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, ScoredPoint
from typing import List
from colorama import init, Fore, Style
init()

def find_similar_paper_ids(collection_name: str, paper_id: str, top_k: int = 5) -> List[str]:
    client = QdrantClient(host="localhost", port=6333)

    # Step 1: Scroll to retrieve the point with a given paper ID from payload
    scroll_filter = Filter(
        must=[
            FieldCondition(
                key="id",  # payload key
                match=MatchValue(value=paper_id)
            )
        ]
    )

    response, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=1,
        with_vectors=True  # needed to get the embedding
    )

    if not response:
        raise ValueError(f"❌ Paper ID '{paper_id}' not found in the collection.")

    query_vector = response[0].vector  # This is the embedding

    # Step 2: Perform semantic search using the query vector
    results: List[ScoredPoint] = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        # offset=1,
        with_payload=True,
        with_vectors=False
    ).points

    # Step 3: Extract and return paper IDs from the results
    similar_ids = [point.payload.get("id") for point in results if point.payload and "id" in point.payload]

    return similar_ids

# Put main statement here
#     collection = "arxiv_papers"
#     paper_id = "1311.5068"
#
#     similar_papers = find_similar_paper_ids(collection, paper_id, top_k=5)
#     # print("Top similar paper IDs:", similar_papers)
#     # print(f"{Fore.BLUE} Top similar papers IDs: {similar_papers}")
#     print(similar_papers)
