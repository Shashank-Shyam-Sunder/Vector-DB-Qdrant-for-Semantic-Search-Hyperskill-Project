import os
import time
import json
import matplotlib.pyplot as plt
from typing import List, Dict
from pprint import pprint
from statistics import mean
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint, SearchParams, HnswConfigDiff
from colorama import init, Fore

init()

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = 'arxiv_papers'
RUNS = 10
QUERY_FILE = "test_queries_embeddings.json"
OUTPUT_DIR = "qdrant_runs"

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=600)

with open(QUERY_FILE, 'r', encoding='utf-8') as f:
    test_dataset = json.load(f)

configs = [
    {"m": 8, "ef_construct": 50},
    {"m": 8, "ef_construct": 100},
    {"m": 16, "ef_construct": 32},
    {"m": 16, "ef_construct": 50},
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def wait_for_collection_green(client_name, collection_name, poll_interval=60, max_wait_minutes=90):
    print(f"{Fore.CYAN}⏳ Waiting for collection '{collection_name}' to become green...")
    start_time = time.time()
    timeout_secs = max_wait_minutes * 60

    while True:
        status = client_name.get_collection(collection_name).status
        elapsed = time.time() - start_time

        if status == "green":
            print(f"{Fore.GREEN}✅ Collection is green. Wait time: {int(elapsed)}s")
            break
        elif elapsed >= timeout_secs:
            raise TimeoutError(f"{Fore.RED}❌ Timeout: Collection not green in {max_wait_minutes} minutes")
        else:
            print(f"{Fore.YELLOW}🟡 Optimizing... Elapsed: {int(elapsed)}s | Next check in {poll_interval}s")
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
        ann_times.append(time.time() - start_time_ann)

        knn_result: List[ScoredPoint] = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            search_params=SearchParams(exact=True),
        ).points

        ann_ids = {pt.id for pt in ann_result}
        knn_ids = {pt.id for pt in knn_result}
        precision_list.append(len(ann_ids.intersection(knn_ids)) / top_k)

    return round(mean(precision_list), 4), round(mean(ann_times), 4)


def run_benchmark():
    all_runs = []
    for run in range(1, RUNS + 1):
        print(f"{Fore.MAGENTA}\n🚀 Starting Run {run}...")
        run_results = []
        for cfg in configs:
            print(f"{Fore.BLUE}🔧 Config: m={cfg['m']}, ef_construct={cfg['ef_construct']}")
            client.update_collection(
                collection_name=COLLECTION_NAME,
                hnsw_config=HnswConfigDiff(**cfg)
            )
            wait_for_collection_green(client, COLLECTION_NAME)

            precision, ann_time = get_search_results_precision_and_times(test_dataset, top_k=10)
            run_results.append(
                {"m": cfg["m"], "ef_construct": cfg["ef_construct"], "precision": precision, "ann_time": ann_time})

        run_path = os.path.join(OUTPUT_DIR, f"run_{run}.json")
        with open(run_path, "w", encoding="utf-8") as f:
            json.dump(run_results, f, indent=4)
        all_runs.append(run_results)

    # Averaging the results
    averaged_results = []
    for i in range(len(configs)):
        avg_precision = round(mean([run[i]["precision"] for run in all_runs]), 4)
        avg_ann_time = round(mean([run[i]["ann_time"] for run in all_runs]), 4)
        averaged_results.append({**configs[i], "precision": avg_precision, "ann_time": avg_ann_time})

    with open(os.path.join(OUTPUT_DIR, "averaged_results.json"), "w", encoding="utf-8") as f:
        json.dump(averaged_results, f, indent=4)

    pprint(averaged_results)
    print(f"{Fore.GREEN}\n📁 Results written to '{OUTPUT_DIR}'")

    # Plot
    m_all = [res["m"] for res in averaged_results]
    ef_all = [res["ef_construct"] for res in averaged_results]
    p_all = [res["precision"] for res in averaged_results]
    t_all = [res["ann_time"] for res in averaged_results]

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.plot(m_all, p_all, marker='o', color='r');
    ax1.set_title("Precision vs m")
    ax2.plot(m_all, t_all, marker='+', color='b');
    ax2.set_title("Time vs m")
    ax3.plot(ef_all, p_all, marker='x', color='g');
    ax3.set_title("Precision vs ef_construct")
    ax4.plot(ef_all, t_all, marker='s', color='purple');
    ax4.set_title("Time vs ef_construct")

    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "averaged_results_plot.png")
    plt.savefig(plot_path)
    print(f"{Fore.CYAN}📊 Plot saved to '{plot_path}'")
    plt.show()


if __name__ == '__main__':
    run_benchmark()
