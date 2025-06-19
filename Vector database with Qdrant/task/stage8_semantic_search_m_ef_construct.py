from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint, SearchParams, HnswConfigDiff
from typing import List, Dict, Any
import time
import json
from colorama import init, Fore
from pprint import pprint
from itertools import islice
init()
import matplotlib.pyplot as plt

### STEP: Loading the Qdrant Vector Database
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

### STEP: Loading Query files to test the precision on 100 queries
QUERIES_FILE = r"test_queries_embeddings.json"
with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
    test_dataset = json.load(file)

def get_search_results_precision_and_times(query_dict: Dict[str,float], top_k: int = 10) -> tuple[
    float, float]:
    ann_times = [] # Approximate
    knn_times = [] # Exact
    precision_list = []

    for query, query_embedding in query_dict.items():
        # print(f'{Fore.YELLOW} Query: {query}')

        start_time_ann = time.time()
        ann_result: List[ScoredPoint] = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=top_k
            ).points

        ann_time = time.time() - start_time_ann
        ann_times.append(ann_time)

        start_time_knn = time.time()
        knn_result: List[ScoredPoint] = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            search_params=SearchParams(exact=True),
        ).points

        knn_time = time.time() - start_time_knn
        knn_times.append(knn_time)

        ann_ids = set(item.id for item in ann_result)
        knn_ids = set(item.id for item in knn_result)
        precision = len(ann_ids.intersection(knn_ids)) / top_k

        precision_list.append(precision)

    avg_precision = round(sum(precision_list) / len(precision_list), 4)
    avg_ann_time = round(sum(ann_times) / len(ann_times), 4)

    return avg_precision, avg_ann_time

### STEP: Setting different configuration parameters to update Qdrant collection
configs = [
    {"m": 8, "ef_construct": 50},
    {"m": 8, "ef_construct": 100},
    {"m": 16, "ef_construct": 32},
    {"m": 16, "ef_construct": 50},
]

result_precision_all = []
result_ann_time_all = []

for cfg in configs:
    client.update_collection(
        collection_name=COLLECTION_NAME,
        hnsw_config=HnswConfigDiff(**cfg)
    )

    result_precision, result_ann_time = get_search_results_precision_and_times(test_dataset, top_k=10)

    result_precision_all.append(result_precision)
    result_ann_time_all.append(result_ann_time)


# Create figure and axes
fig = plt.figure(figsize=(10, 6))  # figsize is optional, in inches (width, height)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

m_all = []
ef_construct_all = []
for cfg in configs:
    m_all.append(cfg["m"])
    ef_construct_all.append(cfg["ef_construct"])

ax1.plot(m_all, result_precision_all, ls='-', lw=2, marker='o', color="r")
ax2.plot(m_all, result_ann_time_all, ls='--', marker='+', color="b")
ax3.plot(ef_construct_all, result_ann_time_all, ls='--', marker='+', color="b")
ax4.plot(ef_construct_all, result_ann_time_all, ls='--', marker='+', color="b")
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)
ax1.set_xlabel('m')
ax2.set_xlabel('m')
ax3.set_xlabel('ef_construct')
ax4.set_xlabel('ef_construct')
ax1.set_ylabel('Precision')
ax2.set_ylabel('ANN Query time')
ax3.set_ylabel('Precision')
ax4.set_ylabel('ANN Query time')
fig.show()

