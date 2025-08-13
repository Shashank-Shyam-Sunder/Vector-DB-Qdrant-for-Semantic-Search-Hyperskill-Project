from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint, SearchParams
from typing import List, Dict, Any
import time
import json
from colorama import init, Fore
from itertools import islice
init()

### STEP: Loading the Qdrant Vector Database
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

### STEP: Loading Query files to test the precision on 100 queries
QUERIES_FILE = r"test_queries_embeddings.json"
with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
    test_dataset = json.load(file)

# print(f'{Fore.GREEN} {test_dataset}')
# %%
### STEP: Defining the function for approximate ANN and exact KNN neighbours search

def get_search_results_precision_and_times(query_dict: Dict[str,float], top_k: int = 10) -> tuple[
    list[Any], list[Any], list[Any]]:
    ann_times = [] # Approximate
    knn_times = [] # Exact
    precision_list = []

    for query, query_embedding in query_dict.items():
        print(f'{Fore.YELLOW} Query: {query}')

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

    return precision_list, ann_times, knn_times

# %%
# test_set = test_dataset.copy()
# test_set = dict(islice(test_dataset.items(), 5))

# precision_list_all, ann_times_all, knn_times_all = get_search_results_precision_and_times(test_set)
precision_list_all, ann_times_all, knn_times_all = get_search_results_precision_and_times(test_dataset, top_k=20)

def result_formatting(precision_list, ann_times, knn_times):
    avg_precision = sum(precision_list) / len(precision_list)
    avg_ann_time = sum(ann_times) / len(ann_times)
    avg_knn_time = sum(knn_times) / len(knn_times)
    print(f'{Fore.BLUE} Average precision@10: {avg_precision:.4f}')
    print(f'{Fore.GREEN} Average ANN query time: {avg_ann_time * 1000:.2f} ms')
    print(f'{Fore.MAGENTA} Average exact k-NN query time: {avg_knn_time * 1000:.2f} ms')

print()
result_formatting(precision_list_all, ann_times_all, knn_times_all)
# %%