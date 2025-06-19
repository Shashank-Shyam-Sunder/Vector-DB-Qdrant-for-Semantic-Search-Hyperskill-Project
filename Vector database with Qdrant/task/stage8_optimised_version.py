from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint, SearchParams, HnswConfigDiff
from typing import List, Dict
import time
import json
from colorama import init, Fore
import matplotlib.pyplot as plt
from pprint import pprint
init()

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'

client = QdrantClient(host=QDRANT_HOST, port=6333, timeout=600)

QUERIES_FILE = r"test_queries_embeddings.json"
with open(QUERIES_FILE, 'r', encoding='utf-8') as file:
    test_dataset = json.load(file)


def wait_for_collection_green(client_name, collection_name, poll_interval=60, max_wait_minutes=90):
    """
    Wait until the Qdrant collection becomes 'green' (fully optimized).

    Args:
        client_name: QdrantClient instance
        collection_name (str): Name of the collection to monitor
        poll_interval (int): Seconds between status checks (default: 60s)
        max_wait_minutes (int): Maximum time to wait (default: 90 minutes)

    Raises:
        TimeoutError: If collection doesn't become green within the max wait time
    """
    print(f"{Fore.CYAN}⏳ Waiting for collection '{collection_name}' to become green...")
    start_time = time.time()
    timeout_secs = max_wait_minutes * 60

    while True:
        status = client_name.get_collection(collection_name).status
        elapsed = time.time() - start_time

        if status == "green":
            print(
                f"{Fore.GREEN}✅ Collection '{collection_name}' is green and ready. Total wait time: {int(elapsed)} seconds.")
            break
        elif elapsed >= timeout_secs:
            raise TimeoutError(
                f"{Fore.RED}❌ Timeout: Collection '{collection_name}' did not reach 'green' status within {max_wait_minutes} minutes."
            )
        else:
            print(
                f"{Fore.YELLOW}🟡 Still optimizing... Status: {status} | Elapsed: {int(elapsed)}s | Next check in {poll_interval}s")
            time.sleep(poll_interval)

def get_search_results_precision_and_times(query_dict: Dict[str, List[float]], top_k: int = 10):
    ann_times, precision_list = [], []

    for query, query_embedding in query_dict.items():
        start_time_ann = time.time()
        ann_result: List[ScoredPoint] = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k
        ).points
        ann_time = time.time() - start_time_ann
        ann_times.append(ann_time)

        knn_result: List[ScoredPoint] = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            search_params=SearchParams(exact=True),
        ).points

        ann_ids = set(item.id for item in ann_result)
        knn_ids = set(item.id for item in knn_result)
        precision = len(ann_ids.intersection(knn_ids)) / top_k
        precision_list.append(precision)

    return round(sum(precision_list) / len(precision_list), 4), round(sum(ann_times) / len(ann_times), 4)

# configs = [
#     {"m": 8, "ef_construct": 50},
#     {"m": 8, "ef_construct": 100},
#     {"m": 16, "ef_construct": 32},
#     {"m": 16, "ef_construct": 50},
# ]
configs = [
    {"m": 16, "ef_construct": 100}
]

result_precision_all, result_ann_time_all = [], []
m_all, ef_construct_all = [], []

for cfg in configs:
    print(f"{Fore.MAGENTA}🔧 Updating HNSW config: m={cfg['m']}, ef_construct={cfg['ef_construct']}")
    client.update_collection(
        collection_name=COLLECTION_NAME,
        hnsw_config=HnswConfigDiff(**cfg)
    )
    wait_for_collection_green(client, COLLECTION_NAME)

    avg_precision, avg_ann_time = get_search_results_precision_and_times(test_dataset, top_k=10)
    result_precision_all.append(avg_precision)
    result_ann_time_all.append(avg_ann_time)
    m_all.append(cfg["m"])
    ef_construct_all.append(cfg["ef_construct"])

# Combine results into structured output
results_summary = []
for i in range(len(configs)):
    results_summary.append({
        "m": m_all[i],
        "ef_construct": ef_construct_all[i],
        "precision": result_precision_all[i],
        "ann_time": result_ann_time_all[i]
    })

# Print result in a pretty JSON format
print("\nm and ef_construct TEST RESULTS (json.dumps):")
print(json.dumps(results_summary, indent=4))

# Print result in pprint style
print("\nm and ef_construct (pprint):")
pprint(results_summary)

save_results = False
if save_results:
    # Export results to a JSON file
    output_json_path = "m_and_ef_construct_test_results2.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4)

    print(f"\n📁 Results successfully written to '{output_json_path}'")

    # Plot results
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.plot(m_all, result_precision_all, ls='-', marker='o', color='r')
    ax2.plot(m_all, result_ann_time_all, ls='--', marker='+', color='b')
    ax3.plot(ef_construct_all, result_precision_all, ls='-', marker='x', color='g')
    ax4.plot(ef_construct_all, result_ann_time_all, ls='--', marker='s', color='purple')

    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(True)

    ax1.set_xlabel('m');
    ax1.set_ylabel('Precision')
    ax2.set_xlabel('m');
    ax2.set_ylabel('ANN Query Time (s)')
    ax3.set_xlabel('ef_construct');
    ax3.set_ylabel('Precision')
    ax4.set_xlabel('ef_construct');
    ax4.set_ylabel('ANN Query Time (s)')
    fig.tight_layout()
    fig.savefig('m_and_ef_construct_test_results2.png')
    fig.show()


