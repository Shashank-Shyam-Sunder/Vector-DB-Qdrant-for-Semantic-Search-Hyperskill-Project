from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint, SearchParams
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

collection_info = client.get_collection(collection_name=COLLECTION_NAME)
print(json.dumps(collection_info.model_dump(), indent=4))

# %%
### STEP: Loading Query files to test the precision on 100 queries
QUERIES_FILE = r"test_queries_embeddings.json"
with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
    test_dataset = json.load(file)

def get_search_results_precision_and_times(query_dict: Dict[str,float], ef_hnsw: int = None, top_k: int = 10) -> dict[str,Any]:
    ann_times = [] # Approximate
    knn_times = [] # Exact
    precision_list = []

    for query, query_embedding in query_dict.items():
        # print(f'{Fore.YELLOW} Query: {query}')

        start_time_ann = time.time()
        ann_result: List[ScoredPoint] = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=top_k,
                search_params=SearchParams(hnsw_ef=ef_hnsw)
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

    avg_precision = round(sum(precision_list) / len(precision_list),4)
    avg_ann_time = round(sum(ann_times) / len(ann_times),4)

    result_dict = {"ef_hnsw": ef_hnsw, "precision": avg_precision, "ann_time": avg_ann_time}
    # print(result_dict)
    return result_dict

ef_hnsw_test_list = [10, 20, 50, 100, 200]
# ef_hnsw_test_list = [10,100]

results_all = []
for ef_hnsw_test in ef_hnsw_test_list:
    results = get_search_results_precision_and_times(test_dataset, ef_hnsw=ef_hnsw_test, top_k=10)
    results_all.append(results)

# pprint(f'EF HNSW TEST RESULTS: {results_all}',indent=100)
print(Fore.GREEN + 'EF HNSW TEST RESULTS:')
print(Fore.LIGHTRED_EX + json.dumps(results_all, indent=4))
print()
print(Fore.LIGHTMAGENTA_EX)
pprint(results_all, indent=1)

# %%
# Create figure and axes
plt.close('all')
fig = plt.figure(figsize=(10, 6))  # figsize is optional, in inches (width, height)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

precision_all = []
ann_time_all = []
for result in results_all:
    precision_all.append(result["precision"])
    ann_time_all.append(result["ann_time"])

ax1.plot(ef_hnsw_test_list, precision_all, ls='-', lw=2, marker='o', color="r")
ax2.plot(ef_hnsw_test_list, ann_time_all, ls='--', marker='+', color="b")
ax1.grid(True)
ax2.grid(True)
# ax1.set_xlabel('ef_hnsw')
ax2.set_xlabel('ef_hnsw')
ax1.set_ylabel('Precision')
ax2.set_ylabel('ANN Query time')
fig.show()

