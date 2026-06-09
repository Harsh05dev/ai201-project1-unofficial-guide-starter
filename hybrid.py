"""Stretch 2 — Hybrid search: semantic (dense) + BM25 (lexical), fused with RRF.

Dense embeddings capture meaning but under-weight exact tokens like course codes
("CS644") and proper nouns ("Hadoop"); BM25 nails those exact matches but misses
paraphrase. Reciprocal Rank Fusion (RRF) combines both rankings so a chunk that
ranks well in either method surfaces.

RRF score for a chunk = sum over methods of 1 / (rrf_k + rank), rrf_k = 60.
"""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from ingest import build_chunks
from retriever import get_model, _client
from config import CHROMA_COLLECTION, N_RESULTS

_chunks = None
_bm25 = None


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _load():
    """Build (once) the chunk list + BM25 index, aligned with ChromaDB's chunk-<i> ids."""
    global _chunks, _bm25
    if _chunks is None:
        _chunks = build_chunks()
        _bm25 = BM25Okapi([_tokenize(c["text"]) for c in _chunks])
    return _chunks, _bm25


def _semantic_ranking(query, n):
    """Top-n chunk indices by dense similarity (parsed from ChromaDB 'chunk-<i>' ids)."""
    collection = _client().get_collection(CHROMA_COLLECTION)
    q_emb = get_model().encode([query]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=n)
    return [int(cid.split("-")[1]) for cid in res["ids"][0]]


def _bm25_ranking(query, n):
    """Top-n chunk indices by BM25 lexical score."""
    _, bm25 = _load()
    scores = bm25.get_scores(_tokenize(query))
    return list(np.argsort(scores)[::-1][:n])


def hybrid_retrieve(query, k=N_RESULTS, candidates=25, rrf_k=60):
    """Return top-k chunks by Reciprocal Rank Fusion of semantic + BM25 rankings."""
    chunks, _ = _load()
    sem = _semantic_ranking(query, min(candidates, len(chunks)))
    bm = _bm25_ranking(query, min(candidates, len(chunks)))

    rrf = {}
    for rank, idx in enumerate(sem):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, idx in enumerate(bm):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

    top = sorted(rrf, key=rrf.get, reverse=True)[:k]
    results = []
    for idx in top:
        meta = chunks[idx]["metadata"]
        results.append({
            "text": chunks[idx]["text"],
            "source": meta["source"],
            "professor": meta["professor"],
            "course": meta["course"],
            "rating": meta.get("rating", 0.0),
            "distance": None,            # fused result — rank-based, no single distance
            "rrf_score": rrf[idx],
        })
    return results


if __name__ == "__main__":
    # Compare semantic-only vs hybrid on keyword-heavy queries (Stretch 2 report).
    from retriever import ensure_index, retrieve
    ensure_index()
    for q in [
        "Hadoop big data",                 # exact keyword BM25 should boost
        "CS644",                           # exact course code
        "bucket graded can still fail",    # paraphrase, semantic-friendly
    ]:
        print("=" * 78)
        print(f"Query: {q}\n")
        print("  SEMANTIC-ONLY top 3:")
        for r in retrieve(q, k=3):
            print(f"    [{r['distance']:.3f}] {r['professor']} ({r['course']})")
        print("  HYBRID (semantic + BM25) top 3:")
        for r in hybrid_retrieve(q, k=3):
            print(f"    [rrf {r['rrf_score']:.4f}] {r['professor']} ({r['course']})")
        print()
