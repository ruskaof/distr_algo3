import numpy as np

from common import (
    build_hnsw, build_index_set, build_ivfpq, build_lsh,
    load_data, sq_euclidean,
)

INDEX_SIZE = 500_000
TOP_K      = 100
SEED       = 42

HNSW_CFG  = {"m": 12, "ef_construction": 80, "ef_query": 80}
LSH_CFG   = {"nbits": 1024}
IVFPQ_CFG = {"nlist": 512, "m_pq": 15, "nbits": 8, "nprobe": 64}


def word(corpus_words, index_corpus_ids, local_pos: int) -> str:
    return str(corpus_words[index_corpus_ids[local_pos]]).replace("_NOUN", "")


def show_hits_and_misses(
    label: str,
    approx: list[int],
    exact: np.ndarray,
    query_vec: np.ndarray,
    index_vectors: np.ndarray,
    index_corpus_ids: np.ndarray,
    corpus_words: np.ndarray,
    n: int = 5,
) -> None:
    approx_set = set(approx)
    exact_list = exact.tolist()

    hits   = [p for p in exact_list if p in approx_set][:n]
    misses = [p for p in exact_list if p not in approx_set][:n]

    print(f"\n  {label}")
    print(f"  hits ({len(hits)}/{n}):")
    for rank, lp in enumerate(hits, 1):
        dist = sq_euclidean(query_vec, index_vectors[lp])
        print(f"    {rank}. {word(corpus_words, index_corpus_ids, lp):<30}  dist={dist:.4f}")

    print(f"  missed ({len(misses)}/{n}):")
    for rank, lp in enumerate(misses, 1):
        dist = sq_euclidean(query_vec, index_vectors[lp])
        print(f"    {rank}. {word(corpus_words, index_corpus_ids, lp):<30}  dist={dist:.4f}")


def main():
    rng = np.random.default_rng(SEED)

    corpus_vectors, corpus_words, query_indices, ground_truth = load_data()
    print(f"{len(corpus_vectors)} nouns, {len(query_indices)} queries")

    index_vectors, index_corpus_ids, _, query_local_pos, query_vectors, gt_local = build_index_set(
        corpus_vectors, query_indices, INDEX_SIZE, TOP_K, ground_truth, rng
    )
    print(f"index: {len(index_corpus_ids)} vectors\n")

    print("building HNSW ...")
    hnsw = build_hnsw(index_vectors, HNSW_CFG)

    print("building LSH ...")
    lsh = build_lsh(index_vectors, LSH_CFG)

    print("building IVFPQ ...")
    ivfpq = build_ivfpq(index_vectors, IVFPQ_CFG)

    qi    = int(rng.integers(len(query_indices)))
    lp    = int(query_local_pos[qi])
    qv    = query_vectors[qi]
    exact = gt_local[qi]

    print(f"\nquery word: {word(corpus_words, index_corpus_ids, lp)}  "
          f"(#{qi}, {len(exact)} exact neighbours in index)")

    labels, _ = hnsw.knn_query(qv.reshape(1, -1), k=TOP_K + 1)
    hnsw_neighbours = [x for x in labels[0].tolist() if x != lp][:TOP_K]

    _, I = lsh.search(qv.reshape(1, -1), TOP_K + 1)
    lsh_neighbours = [int(x) for x in I[0] if x >= 0 and x != lp][:TOP_K]

    _, I = ivfpq.search(qv.reshape(1, -1), TOP_K + 1)
    ivfpq_neighbours = [int(x) for x in I[0] if x >= 0 and x != lp][:TOP_K]

    show_hits_and_misses("HNSW  (quality config)", hnsw_neighbours, exact,
                         qv, index_vectors, index_corpus_ids, corpus_words)
    show_hits_and_misses("LSH   (quality config)", lsh_neighbours, exact,
                         qv, index_vectors, index_corpus_ids, corpus_words)
    show_hits_and_misses("IVFPQ (quality config)", ivfpq_neighbours, exact,
                         qv, index_vectors, index_corpus_ids, corpus_words)


if __name__ == "__main__":
    main()
