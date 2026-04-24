import os
import time

import numpy as np

NUM_QUERIES = 10_000
TOP_K       = 100
CORPUS_SIZE = "all"
SEED        = 42
BATCH_SIZE  = 500


def load_model(corpus_size: int | None):
    import gensim.downloader as api

    print("loading word2vec-ruscorpora-300 ...")
    model = api.load("word2vec-ruscorpora-300")
    print(f"  vocab: {len(model)} words")

    all_words   = np.array(model.index_to_key, dtype=object) # pyright: ignore[reportAttributeAccessIssue]
    all_vectors = model.vectors.astype(np.float32) # pyright: ignore[reportAttributeAccessIssue]

    noun_mask = np.array([str(w).endswith("_NOUN") for w in all_words])
    words   = all_words[noun_mask]
    vectors = all_vectors[noun_mask]
    print(f"  keeping nouns only: {len(words)} / {len(all_words)}")

    if corpus_size is not None and corpus_size < len(words):
        print(f"  trimming to {corpus_size}")
        words   = words[:corpus_size]
        vectors = vectors[:corpus_size]

    return words, vectors


def sample_queries(n_corpus: int, num_queries: int, rng: np.random.Generator):
    if num_queries > n_corpus:
        raise ValueError(f"num_queries ({num_queries}) > corpus size ({n_corpus})")
    indices = rng.choice(n_corpus, size=num_queries, replace=False)
    indices.sort()
    return indices


def exact_knn_euclidean(corpus: np.ndarray, query_indices: np.ndarray,
                        top_k: int, batch_size: int) -> np.ndarray:
    n_queries = len(query_indices)
    neighbours = np.empty((n_queries, top_k), dtype=np.int64)

    corpus_sq_norms = (corpus * corpus).sum(axis=1)

    print(f"  exact KNN for {n_queries} queries in batches of {batch_size} ...")
    n_batches = (n_queries + batch_size - 1) // batch_size

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_queries)
        q_idx = query_indices[start:end]
        q_vecs = corpus[q_idx]

        q_sq_norms = (q_vecs * q_vecs).sum(axis=1, keepdims=True)
        cross = q_vecs @ corpus.T
        sq_dists = q_sq_norms + corpus_sq_norms - 2.0 * cross

        for i_local, i_corpus in enumerate(q_idx):
            sq_dists[i_local, i_corpus] = np.inf

        top_indices = np.argpartition(sq_dists, top_k, axis=1)[:, :top_k]
        for i_local in range(len(q_idx)):
            top_sorted = top_indices[i_local][
                np.argsort(sq_dists[i_local, top_indices[i_local]])
            ]
            neighbours[start + i_local] = top_sorted

        print(f"    batch {b + 1}/{n_batches}  ({end}/{n_queries} done)")

    print("  done")
    return neighbours


def main():
    corpus_size = None if CORPUS_SIZE == "all" else int(CORPUS_SIZE)
    rng = np.random.default_rng(SEED)

    os.makedirs("data", exist_ok=True)

    words, vectors = load_model(corpus_size)
    n_corpus = len(words)
    print(f"  {n_corpus} vectors, dim={vectors.shape[1]}")

    np.save("data/corpus_vectors.npy", vectors)
    np.save("data/corpus_words.npy", words)
    print("  saved corpus_vectors.npy, corpus_words.npy")

    num_queries = min(NUM_QUERIES, n_corpus)
    query_indices = sample_queries(n_corpus, num_queries, rng)
    np.save("data/query_indices.npy", query_indices)
    print(f"  sampled {num_queries} query words")

    print("\n10 example query words:")
    for i in query_indices[:10]:
        word = words[i]
        vec  = vectors[i]
        preview = ", ".join(f"{v:+.4f}" for v in vec[:6])
        print(f"  {str(word):<35}  [{preview}, ...]  norm={np.linalg.norm(vec):.4f}")

    print(f"\ncomputing ground-truth top-{TOP_K} neighbours ...")
    ground_truth = exact_knn_euclidean(vectors, query_indices, TOP_K, BATCH_SIZE)
    np.save("data/ground_truth.npy", ground_truth)
    print("  saved ground_truth.npy")


if __name__ == "__main__":
    main()
