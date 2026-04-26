import os
import tempfile
from typing import Any

import faiss
import hnswlib
import numpy as np


def sq_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.dot(d, d))


def build_hnsw(vectors: np.ndarray, cfg: dict[str, Any]) -> hnswlib.Index:
    dim = vectors.shape[1]
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=len(vectors),
                   ef_construction=cfg["ef_construction"],
                   M=cfg["m"])
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
    idx = faiss.IndexIVFPQ(quantizer, dim, cfg["nlist"], cfg["m_pq"], cfg["nbits"])
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
    for fname in ("data/corpus_vectors.npy", "data/corpus_words.npy",
                  "data/query_indices.npy", "data/ground_truth.npy"):
        if not os.path.exists(fname):
            raise FileNotFoundError(f"{fname} not found – run 01_prepare_data.py first")
    corpus_vectors = np.load("data/corpus_vectors.npy")
    corpus_words   = np.load("data/corpus_words.npy", allow_pickle=True)
    query_indices  = np.load("data/query_indices.npy")
    ground_truth   = np.load("data/ground_truth.npy")
    return corpus_vectors, corpus_words, query_indices, ground_truth


def build_index_set(
    corpus_vectors: np.ndarray,
    query_indices: np.ndarray,
    index_size: int,
    top_k: int,
    ground_truth: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    n_corpus = len(corpus_vectors)
    index_size = min(index_size, n_corpus)

    query_set = set(query_indices.tolist())
    other_ids = np.array([i for i in range(n_corpus) if i not in query_set])
    n_extra   = min(index_size - len(query_indices), len(other_ids))
    extra_ids = rng.choice(other_ids, size=n_extra, replace=False)
    index_corpus_ids = np.sort(np.concatenate([query_indices, extra_ids]))

    index_vectors = corpus_vectors[index_corpus_ids]

    corpus_to_local = np.full(n_corpus, -1, dtype=np.int64)
    corpus_to_local[index_corpus_ids] = np.arange(len(index_corpus_ids), dtype=np.int64)

    query_local_pos = corpus_to_local[query_indices]
    query_vectors   = index_vectors[query_local_pos]

    gt_local = [
        corpus_to_local[ground_truth[i][corpus_to_local[ground_truth[i]] >= 0]][:top_k]
        for i in range(len(query_indices))
    ]

    return index_vectors, index_corpus_ids, corpus_to_local, query_local_pos, query_vectors, gt_local
