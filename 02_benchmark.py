import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import time
from typing import Any

import faiss
import hnswlib
import matplotlib
matplotlib.use("Agg")

INDEX_SIZE = 500_000   # effectively uses the full corpus (~185k nouns)
TOP_K = 100
GRAPHS_DIR = "graphs"

HNSW_M_VALUES = [2, 4, 6, 8, 10, 12]
HNSW_EF_VALUES = [10, 20, 30, 50, 80]

LSH_NBITS_VALUES = [64, 128, 256, 512, 1024, 2048]

IVFPQ_NLIST_VALUES = [64, 256, 1024]
IVFPQ_NPROBE_VALUES = [1, 4, 16, 64]
IVFPQ_M_PQ_VALUES = [15, 30, 60]
IVFPQ_NBITS_VALUES = [8, 10, 12]


def build_hnsw(vectors: np.ndarray, cfg: dict[str, Any]) -> hnswlib.Index:
    dim = vectors.shape[1]
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=len(vectors),
                   ef_construction=cfg["ef_construction"], M=cfg["m"])
    idx.add_items(vectors)
    idx.set_ef(cfg["ef_query"])
    return idx


def build_lsh(vectors: np.ndarray, cfg: dict[str, Any]) -> faiss.IndexLSH:
    idx = faiss.IndexLSH(vectors.shape[1], cfg["nbits"])
    idx.add(vectors)
    return idx


def build_ivfpq(vectors: np.ndarray, cfg: dict[str, Any]) -> faiss.IndexIVFPQ:
    dim = vectors.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    idx = faiss.IndexIVFPQ(
        quantizer, dim, cfg["nlist"], cfg["m_pq"], cfg["nbits"])
    idx.train(vectors)
    idx.add(vectors)
    idx.nprobe = cfg["nprobe"]
    return idx


def measure_size_mb(index: Any) -> float:
    if isinstance(index, faiss.Index):
        return faiss.serialize_index(index).nbytes / 1024 / 1024
    if isinstance(index, hnswlib.Index):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            index.save_index(path)
            return os.path.getsize(path) / 1024 / 1024
        finally:
            os.unlink(path)
    return 0.0


def recall_at_k(approx: list[int], exact: np.ndarray, k: int) -> float:
    if not approx:
        return 0.0
    return len(set(approx[:k]) & set(exact[:k].tolist())) / k


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for fname in ("data/corpus_vectors.npy", "data/corpus_words.npy", "data/query_indices.npy", "data/ground_truth.npy"):
        if not os.path.exists(fname):
            raise FileNotFoundError(
                f"{fname} not found - run 01_prepare_data.py first")
    corpus_vectors = np.load("data/corpus_vectors.npy")
    corpus_words = np.load("data/corpus_words.npy", allow_pickle=True)
    query_indices = np.load("data/query_indices.npy")
    ground_truth = np.load("data/ground_truth.npy")
    return corpus_vectors, corpus_words, query_indices, ground_truth


def build_index_set(
    corpus_vectors: np.ndarray,
    query_indices: np.ndarray,
    index_size: int,
    top_k: int,
    ground_truth: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    n_corpus = len(corpus_vectors)
    index_size = min(index_size, n_corpus)

    query_set = set(query_indices.tolist())
    other_ids = np.array([i for i in range(n_corpus) if i not in query_set])
    n_extra = min(index_size - len(query_indices), len(other_ids))
    extra_ids = rng.choice(other_ids, size=n_extra, replace=False)
    index_corpus_ids = np.sort(np.concatenate([query_indices, extra_ids]))

    index_vectors = corpus_vectors[index_corpus_ids]

    corpus_to_local = np.full(n_corpus, -1, dtype=np.int64)
    corpus_to_local[index_corpus_ids] = np.arange(
        len(index_corpus_ids), dtype=np.int64)

    query_local_pos = corpus_to_local[query_indices]
    query_vectors = index_vectors[query_local_pos]

    gt_local = [
        corpus_to_local[ground_truth[i]
                        [corpus_to_local[ground_truth[i]] >= 0]][:top_k]
        for i in range(len(query_indices))
    ]
    return index_vectors, index_corpus_ids, corpus_to_local, query_local_pos, query_vectors, gt_local


def evaluate_hnsw(index, query_vectors, query_local_pos, gt_local, k):
    recalls = []
    t0 = time.perf_counter()
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
            neighbours = []
        else:
            neighbours = [x for x in labels[0].tolist() if x !=
                          int(query_local_pos[i])][:k]
        recalls.append(recall_at_k(neighbours, gt_local[i], k))
    elapsed = time.perf_counter() - t0
    return float(np.mean(recalls)), len(query_vectors) / elapsed


def evaluate_faiss(index, query_vectors, query_local_pos, gt_local, k):
    recalls = []
    t0 = time.perf_counter()
    for i, qv in enumerate(query_vectors):
        _, I = index.search(qv.reshape(1, -1), k + 1)
        neighbours = [int(x) for x in I[0] if x >= 0 and x !=
                      int(query_local_pos[i])][:k]
        recalls.append(recall_at_k(neighbours, gt_local[i], k))
    elapsed = time.perf_counter() - t0
    return float(np.mean(recalls)), len(query_vectors) / elapsed


def plot_sweep(x_values, results, x_label, title, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(title, fontsize=13)
    metrics = [
        ("recall",    "Recall"),
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

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # left: recall vs QPS
    ax = axes[0]
    for res, label, color, marker in series:
        recalls = [r["recall"] for r in res]
        qps = [r["query_qps"] for r in res]
        ax.scatter(recalls, qps, label=label, color=color,
                   marker=marker, s=45, alpha=0.8)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Query QPS", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title("Recall vs QPS", fontsize=13)

    # right: recall vs index size
    ax = axes[1]
    for res, label, color, marker in series:
        recalls = [r["recall"] for r in res]
        sizes = [r["size_mb"] for r in res]
        ax.scatter(recalls, sizes, label=label, color=color,
                   marker=marker, s=45, alpha=0.8)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Index size (MB)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title("Recall vs Index Size", fontsize=13)

    # right: recall vs build time
    ax = axes[2]
    for res, label, color, marker in series:
        recalls = [r["recall"] for r in res]
        build_s = [r["build_s"] for r in res]
        ax.scatter(recalls, build_s, label=label, color=color,
                   marker=marker, s=45, alpha=0.8)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Build time (s)", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title("Recall vs Build Time", fontsize=13)

    fig.suptitle("Algorithm comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def print_table(rows, param_key, param_label):
    header = f"  {param_label:<10} {'Build (s)':>10} {'Recall':>11} {'Query QPS':>11} {'Size (MB)':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(f"  {str(r[param_key]):<10} {r['build_s']:>10.2f} {r['recall']:>11.3f}"
              f" {r['query_qps']:>11.0f} {r['size_mb']:>10.1f}")
    print()


def main():
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    rng = np.random.default_rng(1)

    corpus_vectors, _, query_indices, ground_truth = load_data()
    n_corpus, dim = corpus_vectors.shape
    print(
        f"corpus: {n_corpus} × {dim}  |  queries: {len(query_indices)}  |  k={TOP_K}")

    index_vectors, index_corpus_ids, _, query_local_pos, query_vectors, gt_local = build_index_set(
        corpus_vectors, query_indices, INDEX_SIZE, TOP_K, ground_truth, rng
    )
    n_index = len(index_corpus_ids)
    n_extra = n_index - len(query_indices)
    print(
        f"index: {n_index} vectors  ({len(query_indices)} query + {n_extra} random)\n")

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
            recall, qps = evaluate_hnsw(
                idx, query_vectors, query_local_pos, gt_local, TOP_K)
            size_mb = measure_size_mb(idx)
            print(
                f"  m={m:<3} ef_c={ef:<4} ef_q={ef_query:<4}  built={build_s:.1f}s  "
                f"recall={recall:.3f}  qps={qps:.0f}  size={size_mb:.1f}MB"
            )
            hnsw_grid_results.append(
                {"m": m, "ef": ef, "build_s": build_s, "recall": recall,
                    "query_qps": qps, "size_mb": size_mb}
            )

    print("=== LSH: nbits sweep ===")
    lsh_nbits_results = []
    for nbits in LSH_NBITS_VALUES:
        cfg = {"nbits": nbits}
        t0 = time.perf_counter()
        idx = build_lsh(index_vectors, cfg)
        build_s = time.perf_counter() - t0
        recall, qps = evaluate_faiss(
            idx, query_vectors, query_local_pos, gt_local, TOP_K)
        size_mb = measure_size_mb(idx)
        print(
            f"  nbits={nbits:<6}  built={build_s:.2f}s  recall={recall:.3f}  qps={qps:.0f}  size={size_mb:.1f}MB")
        lsh_nbits_results.append({"nbits": nbits, "build_s": build_s, "recall": recall,
                                  "query_qps": qps, "size_mb": size_mb})
    print_table(lsh_nbits_results, "nbits", "nbits")
    plot_sweep(LSH_NBITS_VALUES, lsh_nbits_results, "nbits",
               "LSH: nbits sweep", f"{GRAPHS_DIR}/lsh_nbits_sweep.png")

    print("=== IVF+PQ: full grid (m_pq x nbits x nlist x nprobe) ===")
    ivfpq_grid_results = []
    for m_pq in IVFPQ_M_PQ_VALUES:
        for nbits in IVFPQ_NBITS_VALUES:
            print(f"  -- m_pq={m_pq}, nbits={nbits}")
            for nlist in IVFPQ_NLIST_VALUES:
                base_cfg = {"nlist": nlist, "m_pq": m_pq,
                            "nbits": nbits, "nprobe": 1}
                t0 = time.perf_counter()
                ivfpq_base = build_ivfpq(index_vectors, base_cfg)
                base_build_s = time.perf_counter() - t0
                base_size_mb = measure_size_mb(ivfpq_base)
                print(
                    f"    nlist={nlist:<5} base built={base_build_s:.2f}s  size={base_size_mb:.1f}MB")
                for nprobe in IVFPQ_NPROBE_VALUES:
                    ivfpq_base.nprobe = nprobe
                    recall, qps = evaluate_faiss(
                        ivfpq_base, query_vectors, query_local_pos, gt_local, TOP_K)
                    print(
                        f"      nprobe={nprobe:<4} recall={recall:.3f} qps={qps:.0f}")
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

    print("=== comparison plot ===")
    plot_comparison(hnsw_grid_results, lsh_nbits_results, ivfpq_grid_results,
                    f"{GRAPHS_DIR}/comparison.png")

    print("\ndone.")


if __name__ == "__main__":
    main()
