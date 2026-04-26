import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import (
    build_hnsw, build_index_set, build_ivfpq, build_lsh,
    load_data, measure_size_mb, recall_at_k,
)

INDEX_SIZE = 500_000   # effectively uses the full corpus (~185k nouns)
TOP_K      = 100
SEED       = 42
GRAPHS_DIR = "graphs"

HNSW_M_VALUES  = [2, 4, 6, 8, 10, 12]
HNSW_EF_VALUES = [10, 20, 30, 50, 80]

LSH_NBITS_VALUES = [64, 128, 256, 512, 1024]

IVFPQ_NLIST_VALUES  = [64, 128, 256, 512, 1024]
IVFPQ_NPROBE_VALUES = [1, 2, 4, 8, 16, 32, 64]
IVFPQ_M_PQ_VALUES   = [15, 30]
IVFPQ_NBITS_VALUES  = [8, 10]


def evaluate_hnsw(index, query_vectors, query_local_pos, gt_local, k):
    recalls = []
    t0 = time.perf_counter()
    fallback_hits = 0
    for i, qv in enumerate(query_vectors):
        # Some sparse HNSW configs (low m / ef) can fail for high k.
        # Retry with smaller request sizes so the full grid run does not crash.
        labels = None
        for req_k in (k + 1, k, max(1, k // 2), 1):
            try:
                labels, _ = index.knn_query(qv.reshape(1, -1), k=req_k)
                break
            except RuntimeError:
                continue
        if labels is None:
            fallback_hits += 1
            neighbours = []
        else:
            neighbours = [x for x in labels[0].tolist() if x != int(query_local_pos[i])][:k]
            if len(neighbours) < k:
                fallback_hits += 1
        recalls.append(recall_at_k(neighbours, gt_local[i], k))
    elapsed = time.perf_counter() - t0
    if fallback_hits:
        print(f"    note: hnsw fallback used for {fallback_hits} / {len(query_vectors)} queries")
    return float(np.mean(recalls)), len(query_vectors) / elapsed


def evaluate_faiss(index, query_vectors, query_local_pos, gt_local, k):
    recalls = []
    t0 = time.perf_counter()
    for i, qv in enumerate(query_vectors):
        _, I = index.search(qv.reshape(1, -1), k + 1)
        neighbours = [int(x) for x in I[0] if x >= 0 and x != int(query_local_pos[i])][:k]
        recalls.append(recall_at_k(neighbours, gt_local[i], k))
    elapsed = time.perf_counter() - t0
    return float(np.mean(recalls)), len(query_vectors) / elapsed


def plot_sweep(x_values, results, x_label, title, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(title, fontsize=13)
    metrics = [
        ("recall",    "Recall@100"),
        ("query_qps", "Query QPS"),
        ("build_s",   "Build time (s)"),
        ("size_mb",   "Index size (MB)"),
    ]
    for ax, (key, ylabel) in zip(axes.flat, metrics):
        ax.plot(x_values, [r[key] for r in results], "o-", linewidth=2)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def plot_comparison(hnsw_res, lsh_res, ivfpq_res, path):
    series = [
        (hnsw_res,  "HNSW (full grid)",       "steelblue",   "o"),
        (lsh_res,   "LSH (nbits sweep)",      "darkorange",  "s"),
        (ivfpq_res, "IVF+PQ (full grid)",     "forestgreen", "^"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # left: recall vs QPS
    ax = axes[0]
    for res, label, color, marker in series:
        recalls = [r["recall"] for r in res]
        qps     = [r["query_qps"] for r in res]
        ax.scatter(recalls, qps, label=label, color=color, marker=marker, s=45, alpha=0.8)
    ax.set_xlabel("Recall@100", fontsize=12)
    ax.set_ylabel("Query QPS", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title("Recall vs QPS", fontsize=13)

    # right: recall vs index size
    ax = axes[1]
    for res, label, color, marker in series:
        recalls = [r["recall"] for r in res]
        sizes   = [r["size_mb"] for r in res]
        ax.scatter(recalls, sizes, label=label, color=color, marker=marker, s=45, alpha=0.8)
    ax.set_xlabel("Recall@100", fontsize=12)
    ax.set_ylabel("Index size (MB)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title("Recall vs Index Size", fontsize=13)

    fig.suptitle("Algorithm comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def print_table(rows, param_key, param_label):
    header = f"  {param_label:<10} {'Build (s)':>10} {'Recall@100':>11} {'Query QPS':>11} {'Size (MB)':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(f"  {str(r[param_key]):<10} {r['build_s']:>10.2f} {r['recall']:>11.3f}"
              f" {r['query_qps']:>11.0f} {r['size_mb']:>10.1f}")
    print()


def main():
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    corpus_vectors, _, query_indices, ground_truth = load_data()
    n_corpus, dim = corpus_vectors.shape
    print(f"corpus: {n_corpus} × {dim}  |  queries: {len(query_indices)}  |  k={TOP_K}")

    index_vectors, index_corpus_ids, _, query_local_pos, query_vectors, gt_local = build_index_set(
        corpus_vectors, query_indices, INDEX_SIZE, TOP_K, ground_truth, rng
    )
    n_index = len(index_corpus_ids)
    n_extra = n_index - len(query_indices)
    print(f"index: {n_index} vectors  ({len(query_indices)} query + {n_extra} random)\n")

    # ── HNSW: full (m, ef) grid ──────────────────────────────────────────────────
    print("=== HNSW: full grid (m x ef_construction, ef_query auto >= k+1) ===")
    hnsw_grid_results = []
    for m in HNSW_M_VALUES:
        for ef in HNSW_EF_VALUES:
            # hnswlib cannot always return top-k when ef_query < k+1.
            # We sweep ef_construction and use a valid ef_query for fair recall@k.
            ef_query = max(ef, TOP_K + 1)
            cfg = {"m": m, "ef_construction": ef, "ef_query": ef_query}
            t0 = time.perf_counter()
            idx = build_hnsw(index_vectors, cfg)
            build_s = time.perf_counter() - t0
            recall, qps = evaluate_hnsw(idx, query_vectors, query_local_pos, gt_local, TOP_K)
            size_mb = measure_size_mb(idx)
            print(
                f"  m={m:<3} ef_c={ef:<4} ef_q={ef_query:<4}  built={build_s:.1f}s  "
                f"recall={recall:.3f}  qps={qps:.0f}  size={size_mb:.1f}MB"
            )
            hnsw_grid_results.append(
                {"m": m, "ef": ef, "build_s": build_s, "recall": recall, "query_qps": qps, "size_mb": size_mb}
            )
    # ── LSH: nbits sweep ─────────────────────────────────────────────────────────
    print("=== LSH: nbits sweep ===")
    lsh_nbits_results = []
    for nbits in LSH_NBITS_VALUES:
        cfg = {"nbits": nbits}
        t0 = time.perf_counter()
        idx = build_lsh(index_vectors, cfg)
        build_s = time.perf_counter() - t0
        recall, qps = evaluate_faiss(idx, query_vectors, query_local_pos, gt_local, TOP_K)
        size_mb = measure_size_mb(idx)
        print(f"  nbits={nbits:<6}  built={build_s:.2f}s  recall={recall:.3f}  qps={qps:.0f}  size={size_mb:.1f}MB")
        lsh_nbits_results.append({"nbits": nbits, "build_s": build_s, "recall": recall,
                                   "query_qps": qps, "size_mb": size_mb})
    print_table(lsh_nbits_results, "nbits", "nbits")
    plot_sweep(LSH_NBITS_VALUES, lsh_nbits_results, "nbits",
               "LSH: nbits sweep", f"{GRAPHS_DIR}/lsh_nbits_sweep.png")

    # ── IVF+PQ: full grid ─────────────────────────────────────────────────────────
    print("=== IVF+PQ: full grid (m_pq x nbits x nlist x nprobe) ===")
    ivfpq_grid_results = []
    for m_pq in IVFPQ_M_PQ_VALUES:
        for nbits in IVFPQ_NBITS_VALUES:
            print(f"  -- m_pq={m_pq}, nbits={nbits}")
            for nlist in IVFPQ_NLIST_VALUES:
                base_cfg = {"nlist": nlist, "m_pq": m_pq, "nbits": nbits, "nprobe": 1}
                t0 = time.perf_counter()
                ivfpq_base = build_ivfpq(index_vectors, base_cfg)
                base_build_s = time.perf_counter() - t0
                base_size_mb = measure_size_mb(ivfpq_base)
                print(f"    nlist={nlist:<5} base built={base_build_s:.2f}s  size={base_size_mb:.1f}MB")
                for nprobe in IVFPQ_NPROBE_VALUES:
                    ivfpq_base.nprobe = nprobe
                    recall, qps = evaluate_faiss(ivfpq_base, query_vectors, query_local_pos, gt_local, TOP_K)
                    print(f"      nprobe={nprobe:<4} recall={recall:.3f} qps={qps:.0f}")
                    ivfpq_grid_results.append(
                        {
                            "m_pq": m_pq,
                            "nbits": nbits,
                            "nlist": nlist,
                            "nprobe": nprobe,
                            "build_s": base_build_s,
                            "recall": recall,
                            "query_qps": qps,
                            "size_mb": base_size_mb,
                        }
                    )
    # ── comparison ───────────────────────────────────────────────────────────────
    print("=== comparison plot ===")
    plot_comparison(hnsw_grid_results, lsh_nbits_results, ivfpq_grid_results,
                    f"{GRAPHS_DIR}/comparison.png")

    print("\ndone.")


if __name__ == "__main__":
    main()
