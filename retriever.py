"""Embedding + vector store (ChromaDB) and semantic retrieval.

Embeds chunks locally with all-MiniLM-L6-v2 and stores them in a persistent
ChromaDB collection with source metadata, then retrieves the top-k most similar
chunks for a query.
"""
import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL, N_RESULTS
from ingest import build_chunks

# Load the embedding model once (cached after first download).
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def embed_and_store(chunks=None):
    """Embed all chunks and (re)build the ChromaDB collection from scratch.

    Returns the number of chunks stored.
    """
    if chunks is None:
        chunks = build_chunks()

    client = _client()
    # Rebuild cleanly so re-running never duplicates chunks.
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"chunk-{i}" for i in range(len(chunks))]
    embeddings = get_model().encode(texts, show_progress_bar=False).tolist()

    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    return len(chunks)


def collection_count():
    """Return how many chunks are currently stored (0 if the collection is missing)."""
    try:
        return _client().get_collection(CHROMA_COLLECTION).count()
    except Exception:
        return 0


def ensure_index():
    """Build the index if it does not exist yet. Returns chunk count."""
    count = collection_count()
    if count == 0:
        count = embed_and_store()
    return count


def _build_where(course=None, professor=None, min_rating=None):
    """Build a ChromaDB `where` clause from optional metadata filters (Stretch 1)."""
    clauses = []
    if course:
        clauses.append({"course": {"$eq": course}})
    if professor:
        clauses.append({"professor": {"$eq": professor}})
    if min_rating:
        clauses.append({"rating": {"$gte": float(min_rating)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def retrieve(query, k=N_RESULTS, course=None, professor=None, min_rating=None):
    """Return the top-k most relevant chunks for a query.

    Optional metadata filters (Stretch 1 — metadata filtering) constrain the search
    to a course, a professor, and/or a minimum overall rating before ranking by
    semantic similarity. Each result is a dict:
    {"text", "source", "professor", "course", "rating", "distance"}.
    """
    collection = _client().get_collection(CHROMA_COLLECTION)
    query_embedding = get_model().encode([query]).tolist()
    where = _build_where(course, professor, min_rating)
    res = collection.query(query_embeddings=query_embedding, n_results=k, where=where)

    results = []
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    for text, meta, dist in zip(docs, metas, dists):
        results.append({
            "text": text,
            "source": meta.get("source", "unknown"),
            "professor": meta.get("professor", "unknown"),
            "course": meta.get("course", "N/A"),
            "rating": meta.get("rating", 0.0),
            "distance": dist,
        })
    return results


def list_courses():
    """Distinct course codes in the index (for the UI filter dropdown)."""
    try:
        metas = _client().get_collection(CHROMA_COLLECTION).get()["metadatas"]
    except Exception:
        return []
    courses = {m.get("course", "N/A") for m in metas if m.get("course", "N/A") != "N/A"}
    return sorted(courses)


def list_professors():
    """Distinct professor names in the index (for the UI filter dropdown)."""
    try:
        metas = _client().get_collection(CHROMA_COLLECTION).get()["metadatas"]
    except Exception:
        return []
    return sorted({m.get("professor", "unknown") for m in metas})


if __name__ == "__main__":
    # Retrieval smoke test (Milestone 4).
    n = embed_and_store()
    print(f"Stored {n} chunks in ChromaDB.\n")
    for q in [
        "Is Andrew Sohn's CS350 hard, and how is it graded?",
        "Which professor do students most recommend for databases?",
        "What is the main study tip for Joseph Leung's classes?",
    ]:
        print(f"=== Query: {q}")
        for r in retrieve(q):
            print(f"  [{r['distance']:.3f}] {r['professor']} ({r['course']}) <{r['source']}>")
            print(f"        {r['text'].splitlines()[-1][:100]}")
        print()
