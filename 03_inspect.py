import numpy as np

from common import (
    HNSW_CONFIGS, IVFPQ_CONFIGS, LSH_CONFIGS,
    build_hnsw, build_index_set, build_ivfpq, build_lsh,
    load_data, sq_euclidean,
)

INDEX_SIZE = 50_000
TOP_K      = 100
SEED       = 42

HNSW_CFG  = next(c for c in HNSW_CONFIGS   if c["name"] == "hnsw_4")
LSH_CFG   = next(c for c in LSH_CONFIGS    if c["name"] == "lsh_4")
IVFPQ_CFG = next(c for c in IVFPQ_CONFIGS  if c["name"] == "ivfpq_4")


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

    print(f"building HNSW ...")
    hnsw = build_hnsw(index_vectors, HNSW_CFG)

    print(f"building LSH ...")
    lsh = build_lsh(index_vectors, LSH_CFG)

    print(f"building IVFPQ ...")
    ivfpq = build_ivfpq(index_vectors, IVFPQ_CFG)

    qi    = int(rng.integers(len(query_indices)))
    lp    = int(query_local_pos[qi])
    qv    = query_vectors[qi]
    exact = gt_local[qi]

    print(f"\nquery word: {word(corpus_words, index_corpus_ids, lp)}  "
          f"(#{qi}, {len(exact)} exact neighbours in index)")

    hnsw_raw        = hnsw.query(qv, k=TOP_K + 1, ef=HNSW_CFG["ef_query"])
    hnsw_neighbours = [key for key, _ in hnsw_raw if key != lp][:TOP_K]

    _, I = lsh.search(qv.reshape(1, -1), TOP_K + 1)
    lsh_neighbours = [int(x) for x in I[0] if x >= 0 and x != lp][:TOP_K]

    _, I = ivfpq.search(qv.reshape(1, -1), TOP_K + 1)
    ivfpq_neighbours = [int(x) for x in I[0] if x >= 0 and x != lp][:TOP_K]

    show_hits_and_misses("HNSW  (recall config)", hnsw_neighbours, exact,
                         qv, index_vectors, index_corpus_ids, corpus_words)
    show_hits_and_misses("LSH   (recall config)", lsh_neighbours, exact,
                         qv, index_vectors, index_corpus_ids, corpus_words)
    show_hits_and_misses("IVFPQ (recall config)", ivfpq_neighbours, exact,
                         qv, index_vectors, index_corpus_ids, corpus_words)


if __name__ == "__main__":
    main()
