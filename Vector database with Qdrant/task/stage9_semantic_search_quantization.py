from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint, SearchParams, QuantizationSearchParams, OptimizersConfigDiff, ScalarQuantization, ScalarQuantizationConfig, ScalarType
from typing import List, Dict, Any
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

def get_search_results_precision_and_times(query_dict: Dict[str, List[float]], rescore_config: Dict[str,Any], top_k: int = 10):
    ann_times, precision_list = [], []

    for query, query_embedding in query_dict.items():
        start_time_ann = time.time()
        ann_result: List[ScoredPoint] = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            search_params=SearchParams(quantization=QuantizationSearchParams(**rescore_config)),
        ).points
        ann_time = time.time() - start_time_ann
        ann_times.append(ann_time)

        knn_result: List[ScoredPoint] = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            search_params=SearchParams(exact=True, quantization=QuantizationSearchParams(ignore=True)),
        ).points

        ann_ids = set(item.id for item in ann_result)
        knn_ids = set(item.id for item in knn_result)
        precision = len(ann_ids.intersection(knn_ids)) / top_k
        precision_list.append(precision)

    return round(sum(precision_list) / len(precision_list), 4), round(sum(ann_times) / len(ann_times), 4)

rescore_configs = [{"rescore": True, "oversampling": 2},
                  {"rescore": False, "oversampling": 2}]

def update_collection_quantization(client_name, collection_name):
    client_name.update_collection(
        collection_name=collection_name,
        optimizer_config=OptimizersConfigDiff(),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=False,
            ),
        ),
    )

print(f"{Fore.MAGENTA}🔧 Updating collection with scalar quantization with Scalartype=INT8, quantil=0.99 and always_ram=False")

update_collection_quantization(client, COLLECTION_NAME)
wait_for_collection_green(client, COLLECTION_NAME)

result_precision_all, result_ann_time_all = [], []
rescore_all, oversampling_all = [], []
for cfg in rescore_configs:
    print(f"{Fore.BLUE}🔍 Testing config: rescore={cfg['rescore']}, oversampling={cfg['oversampling']}")
    avg_precision, avg_ann_time = get_search_results_precision_and_times(test_dataset, cfg, top_k=10)
    result_precision_all.append(avg_precision)
    result_ann_time_all.append(avg_ann_time)
    rescore_all.append(cfg["rescore"])
    oversampling_all.append(cfg["oversampling"])

# Combine results into structured output
results_summary = []
for i in range(len(rescore_configs)):
    results_summary.append({
        "rescore": rescore_all[i],
        "oversampling": oversampling_all[i],
        "precision": result_precision_all[i],
        "ann_time": result_ann_time_all[i]
    })

# Print result in a pretty JSON format
print("\nRescore Quantization TEST RESULTS (json.dumps):")
print(json.dumps(results_summary, indent=4))

# Print result in pprint style
print("\nRescore Quantization TEST RESULTS (pprint):")
pprint(results_summary)

delta = abs(result_precision_all[0] - result_precision_all[1])
print(f"\n🔁 Precision difference between rescore=True and False: {delta:.4f}")

save_results = True
if save_results:
    # Export results to a JSON file
    output_json_path = "quantization_results_with_and_without_rescoring.json"
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4)

    print(f"\n📁 Results successfully written to '{output_json_path}'")

    # Plot results
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    labels = ["True", "False"]
    x =range(len(labels))

    # Precision Plot
    ax1.bar(x, result_precision_all, color='lightgreen')
    ax1.set_title("Precision (Rescore vs No-Rescore)")
    ax1.set_xlabel("Rescore")
    ax1.set_ylabel("Precision")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ANN Time Plot
    ax2.bar(x, result_ann_time_all, color='skyblue')
    ax2.set_title("ANN Query Time (s)")
    ax2.set_xlabel("Rescore")
    ax2.set_ylabel("Time (s)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(True, linestyle='--', alpha=0.5)

    fig.suptitle("Quantization Performance: Rescore vs No-Rescore (Oversampling = 2)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig("quantization_results_rescore_split_plot.png")
    fig.show()

