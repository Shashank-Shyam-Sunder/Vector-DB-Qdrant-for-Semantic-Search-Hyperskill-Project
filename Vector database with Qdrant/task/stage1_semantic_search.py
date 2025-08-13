from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.http.models import PointStruct
from colorama import init, Fore, Style
import json
import uuid
from typing import Any, Generator, List
import time
init()


# Qdrant connection (local Docker)
client = QdrantClient(host="localhost", port=6333)

# Define collection name and vector settings
COLLECTION_NAME = "arxiv_papers"
VECTOR_SIZE = 1536  # Adjust if your embedding size is different

# Create a collection only if it doesn't already exist
if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

print(f"Collection '{COLLECTION_NAME}' is ready.")

FILE_PATH = r"C:\Shashank_work\arxiv_dataset\ml-arxiv-embeddings.json"
BATCH_SIZE = 500

# Generator to stream line-by-line JSON objects
def stream_json(file_path: str) -> Generator[dict, None, None]:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

# Batch loader function
def load_vectors_to_qdrant(file_path: str, batch_size: int = 500) -> None:
    batch: List[PointStruct] = []
    total_records = 0
    start_time = time.time()

    for record in stream_json(file_path):
        embedding = record.get("embedding")
        if not embedding:
            continue  # Skip if embedding is missing

        # Extract payload (everything except the embedding)
        payload = {k: v for k, v in record.items() if k != "embedding"}

        # Create PointStruct with UUID5 from arXiv ID
        point = PointStruct(
            id=str(uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name=record["id"])),
            vector=embedding,
            payload=payload
        )
        batch.append(point)

        # Insert batch when the threshold reached
        if len(batch) >= batch_size:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            total_records += len(batch)
            elapsed_secs = time.time() - start_time
            minutes = int(elapsed_secs // 60)
            seconds = int(elapsed_secs % 60)
            print(f"{Fore.BLUE} Inserted {total_records} records | Time passed: {minutes:02d}:{seconds:02d} (MM:SS)")
            batch = []

    # Final batch insert
    if batch:
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        total_records += len(batch)
        elapsed_secs = time.time() - start_time
        minutes = int(elapsed_secs // 60)
        seconds = int(elapsed_secs % 60)
        print(f"{Fore.MAGENTA} Inserted final batch, total records: {total_records} | Time passed: {minutes:02d}:{seconds:02d} (MM:SS)")

#     print(f"{Fore.YELLOW}🚀 Starting vector loading...")
#     load_vectors_to_qdrant(FILE_PATH, BATCH_SIZE)
#     print(f"{Fore.GREEN}✅ All vectors loaded successfully!")
