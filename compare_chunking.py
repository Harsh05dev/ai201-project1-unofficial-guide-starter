"""Stretch 3 — Chunking strategy comparison.

Compares the project's per-review chunker against a naive fixed-size chunker
(~500 chars, 100 overlap, ignoring review boundaries) on the same embedding model
and the same eval queries. Metric: for each query, is the top-1 chunk from the
correct professor, and what is its cosine distance? Hypothesis (planning.md): the
per-review chunker wins because it respects opinion boundaries instead of merging a
5-star and 1-star review into one blurred window.
"""
import numpy as np

from ingest import load_documents, build_chunks, _parse_professor, MIN_CHUNK_LEN
from retriever import get_model

# (query, expected source file the top result should come from)
EVAL = [
    ("Is Andrew Sohn's CS350 hard, and how is it graded?", "andrew_sohn.txt"),
    ("Which professor do students most recommend for databases?", "james_geller.txt"),
    ("Do students recommend Cristian Borcea for CS643 Cloud Computing?", "cristian_borcea.txt"),
    ("What's the main study tip for Joseph Leung's classes?", "joseph_leung.txt"),
    ("Is Ali Mili an easy professor?", "ali_mili.txt"),
]


def fixed_size_chunks(size=500, overlap=100):
    """Naive baseline: fixed-width windows over each document, ignoring review boundaries."""
    chunks = []
    for filename, text in load_documents():
        professor = _parse_professor(text)
        step = size - overlap
        for i in range(0, len(text), step):
            piece = text[i:i + size].strip()
            if len(piece) >= MIN_CHUNK_LEN:
                chunks.append({"text": piece, "metadata": {"source": filename, "professor": professor}})
    return chunks


def _embed(chunks):
    model = get_model()
    vecs = model.encode([c["text"] for c in chunks], show_progress_bar=False)
    vecs = np.asarray(vecs, dtype=np.float32)
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)  # normalize for cosine
    return vecs


def evaluate(chunks, label):
    vecs = _embed(chunks)
    model = get_model()
    correct = 0
    distances = []
    print(f"\n=== {label}  ({len(chunks)} chunks) ===")
    for query, expected in EVAL:
        q = model.encode([query])[0].astype(np.float32)
        q /= (np.linalg.norm(q) + 1e-9)
        sims = vecs @ q
        top = int(np.argmax(sims))
        dist = 1.0 - float(sims[top])
        distances.append(dist)
        hit = chunks[top]["metadata"]["source"] == expected
        correct += hit
        print(f"  [{dist:.3f}] {'OK ' if hit else 'MISS'} top1={chunks[top]['metadata']['source']} "
              f"(expected {expected})")
    print(f"  -> top-1 correct: {correct}/{len(EVAL)} | avg top-1 distance: {np.mean(distances):.3f}")
    return correct, float(np.mean(distances))


if __name__ == "__main__":
    per_review = build_chunks()
    fixed = fixed_size_chunks()
    r1 = evaluate(per_review, "PER-REVIEW chunker (project strategy)")
    r2 = evaluate(fixed, "FIXED-SIZE chunker (500 chars / 100 overlap)")
    print("\n" + "=" * 60)
    print(f"Per-review : {r1[0]}/5 correct, avg dist {r1[1]:.3f}")
    print(f"Fixed-size : {r2[0]}/5 correct, avg dist {r2[1]:.3f}")
    winner = "per-review" if (r1[0], -r1[1]) >= (r2[0], -r2[1]) else "fixed-size"
    print(f"Winner: {winner}")
