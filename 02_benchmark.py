import time

import numpy as np

from common import (
    ALL_CONFIG_NAMES, HNSW_CONFIGS, IVFPQ_CONFIGS, LSH_CONFIGS,
    build_hnsw, build_index_set, build_ivfpq, build_lsh,
    load_data, measure_size_mb, recall_at_k,
)

INDEX_SIZE = 50_000
TOP_K      = 100
SEED       = 42
CONFIGS    = None


def evaluate_hnsw(
    index,
    query_vectors: np.ndarray,
    query_local_pos: np.ndarray,
    gt_local: list,
    k: int,
    ef_query: int,
) -> tuple[float, float]:
    recalls = []
    t0 = time.perf_counter()
    for i, qv in enumerate(query_vectors):
        results = index.query(qv, k=k + 1, ef=ef_query)
        neighbour_ids = [key for key, _ in results if key != query_local_pos[i]][:k]
        recalls.append(recall_at_k(neighbour_ids, gt_local[i], k)) # pyright: ignore[reportArgumentType]
    elapsed = time.perf_counter() - t0
    return float(np.mean(recalls)), len(query_vectors) / elapsed


def evaluate_lsh(
    index,
    query_vectors: np.ndarray,
    query_local_pos: np.ndarray,
    gt_local: list,
    k: int,
) -> tuple[float, float]:
    recalls = []
    t0 = time.perf_counter()
    for i, qv in enumerate(query_vectors):
        _, I = index.search(qv.reshape(1, -1), k + 1)
        neighbours = [int(x) for x in I[0] if x >= 0 and x != query_local_pos[i]][:k]
        recalls.append(recall_at_k(neighbours, gt_local[i], k))
    elapsed = time.perf_counter() - t0
    return float(np.mean(recalls)), len(query_vectors) / elapsed


def evaluate_ivfpq(
    index,
    query_vectors: np.ndarray,
    query_local_pos: np.ndarray,
    gt_local: list,
    k: int,
) -> tuple[float, float]:
    recalls = []
    t0 = time.perf_counter()
    for i, qv in enumerate(query_vectors):
        q = qv.reshape(1, -1)
        _, I = index.search(q, k + 1)
        neighbours = [int(x) for x in I[0] if x >= 0 and x != query_local_pos[i]][:k]
        recalls.append(recall_at_k(neighbours, gt_local[i], k))
    elapsed = time.perf_counter() - t0
    return float(np.mean(recalls)), len(query_vectors) / elapsed


def print_table(rows: list[dict]) -> None:
    header = f"{'Config':<20} {'Type':<6} {'Index (s)':>10} {'Build QPS':>10} {'Recall K':>10} {'Query QPS':>10} {'Size (MB)':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<20} {r['type']:<6} "
            f"{r['build_s']:>10.1f} {r['index_qps']:>10.1f} "
            f"{r['recall']:>10.3f} {r['query_qps']:>10.1f} "
            f"{r['size_mb']:>10.1f}"
        )
    print("=" * len(header))


def main():
    rng = np.random.default_rng(SEED)

    corpus_vectors, _, query_indices, ground_truth = load_data()
    n_corpus, dim = corpus_vectors.shape
    n_queries = len(query_indices)
    print(f"corpus: {n_corpus} * {dim}  |  queries: {n_queries}  |  k={TOP_K}")

    index_vectors, index_corpus_ids, _, query_local_pos, query_vectors, gt_local = build_index_set(
        corpus_vectors, query_indices, INDEX_SIZE, TOP_K, ground_truth, rng
    )
    n_extra = len(index_corpus_ids) - len(query_indices)
    print(f"index: {len(index_corpus_ids)} vectors  ({len(query_indices)} query + {n_extra} random)")
    print(f"avg in-index ground truth neighbours: {np.mean([len(g) for g in gt_local]):.1f}/{TOP_K}\n")

    selected    = set(CONFIGS) if CONFIGS else set(ALL_CONFIG_NAMES)
    hnsw_cfgs   = [c for c in HNSW_CONFIGS   if c["name"] in selected]
    lsh_cfgs    = [c for c in LSH_CONFIGS    if c["name"] in selected]
    ivfpq_cfgs  = [c for c in IVFPQ_CONFIGS  if c["name"] in selected]
    results     = []

    for cfg in hnsw_cfgs:
        print(f"{cfg['name']}  (m={cfg['m']}, ef_construction={cfg['ef_construction']}, ef_query={cfg['ef_query']})")
        t0 = time.perf_counter()
        idx = build_hnsw(index_vectors, cfg)
        build_s = time.perf_counter() - t0
        print(f"  built in {build_s:.1f}s", end="  ")

        recall, qps = evaluate_hnsw(idx, query_vectors, query_local_pos, gt_local, TOP_K, cfg["ef_query"])
        size_mb = measure_size_mb(idx)
        print(f"recall={recall:.3f}  qps={qps:.0f}  size={size_mb:.1f}MB")
        results.append({"name": cfg["name"], "type": "HNSW", "build_s": build_s,
                        "index_qps": len(index_corpus_ids) / build_s,
                        "recall": recall, "query_qps": qps, "size_mb": size_mb, "k": TOP_K})

    for cfg in lsh_cfgs:
        print(f"{cfg['name']}  (nbits={cfg['nbits']})")
        t0 = time.perf_counter()
        idx = build_lsh(index_vectors, cfg)
        build_s = time.perf_counter() - t0
        print(f"  built in {build_s:.2f}s", end="  ")

        recall, qps = evaluate_lsh(idx, query_vectors, query_local_pos, gt_local, TOP_K)
        size_mb = measure_size_mb(idx)
        print(f"recall={recall:.3f}  qps={qps:.0f}  size={size_mb:.1f}MB")
        results.append({"name": cfg["name"], "type": "LSH", "build_s": build_s,
                        "index_qps": len(index_corpus_ids) / build_s,
                        "recall": recall, "query_qps": qps, "size_mb": size_mb, "k": TOP_K})

    for cfg in ivfpq_cfgs:
        print(f"{cfg['name']}  (nlist={cfg['nlist']}, m_pq={cfg['m_pq']}, nprobe={cfg['nprobe']})")
        t0 = time.perf_counter()
        idx = build_ivfpq(index_vectors, cfg)
        build_s = time.perf_counter() - t0
        print(f"  built in {build_s:.2f}s", end="  ")

        recall, qps = evaluate_ivfpq(idx, query_vectors, query_local_pos, gt_local, TOP_K)
        size_mb = measure_size_mb(idx)
        print(f"recall={recall:.3f}  qps={qps:.0f}  size={size_mb:.1f}MB")
        results.append({"name": cfg["name"], "type": "IVFPQ", "build_s": build_s,
                        "index_qps": len(index_corpus_ids) / build_s,
                        "recall": recall, "query_qps": qps, "size_mb": size_mb, "k": TOP_K})

    if results:
        print_table(results)


if __name__ == "__main__":
    main()
