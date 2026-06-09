"""Document ingestion + chunking.

Strategy (see planning.md): one chunk per student review, plus one profile-summary
chunk per professor. Every chunk is prefixed with a `Professor <Name> (<course>)`
context header so it is self-contained and attributable on its own. No overlap —
review boundaries fall between independent opinions, so bleeding text across them
would only contaminate the embeddings.
"""
import os
import re
import glob

from config import DOCS_PATH

# Minimum characters for a chunk to be kept (drops empty/fragment chunks).
MIN_CHUNK_LEN = 15


def load_documents(docs_path=DOCS_PATH):
    """Load every .txt file in docs_path. Returns list of (filename, text)."""
    documents = []
    for path in sorted(glob.glob(os.path.join(docs_path, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append((os.path.basename(path), text))
    return documents


def _parse_professor(header_text):
    """Pull the professor name out of the 'Professor: <name>' header line."""
    match = re.search(r"Professor:\s*(.+)", header_text)
    return match.group(1).strip() if match else "Unknown"


def _parse_course(bracket_line):
    """Pull a course code (e.g. CS350) out of a '[Course CS350 | ... ]' review header."""
    match = re.search(r"Course\s+([A-Za-z]+\s?\d+\w*)", bracket_line)
    return match.group(1).strip() if match else "N/A"


def chunk_document(filename, text):
    """Split one professor's document into chunks with metadata.

    Returns a list of dicts: {"text": str, "metadata": {...}}.
    """
    chunks = []

    # Split header (profile) from the reviews body.
    if "Student reviews:" in text:
        header, body = text.split("Student reviews:", 1)
    else:
        header, body = text, ""

    professor = _parse_professor(header)

    # 1) Profile-summary chunk (rating, would-take-again, difficulty, courses).
    summary = header.strip()
    if len(summary) >= MIN_CHUNK_LEN:
        chunks.append({
            "text": f"Professor {professor} — profile summary.\n{summary}",
            "metadata": {
                "source": filename,
                "professor": professor,
                "course": "N/A",
                "type": "summary",
            },
        })

    # 2) One chunk per review. Reviews are separated by blank lines; each starts
    #    with a '[Course ... ]' header line followed by the quoted comment.
    blocks = [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]
    for block in blocks:
        lines = block.split("\n")
        bracket = lines[0].strip()
        comment = " ".join(line.strip() for line in lines[1:]).strip()
        if not bracket.startswith("["):
            # Not a recognizable review block; skip stray text.
            continue
        course = _parse_course(bracket)
        meta_info = bracket.strip("[]")
        chunk_text = f"Professor {professor} ({course}) — student review.\n{meta_info}\n{comment}"
        if len(chunk_text) >= MIN_CHUNK_LEN:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": filename,
                    "professor": professor,
                    "course": course,
                    "type": "review",
                },
            })

    return chunks


def build_chunks(docs_path=DOCS_PATH):
    """Load all documents and return a flat list of chunk dicts for the whole corpus."""
    all_chunks = []
    for filename, text in load_documents(docs_path):
        all_chunks.extend(chunk_document(filename, text))
    return all_chunks


if __name__ == "__main__":
    # Inspection step (Milestone 3): print stats and a few sample chunks.
    docs = load_documents()
    chunks = build_chunks()
    print(f"Loaded {len(docs)} documents.")
    print(f"Produced {len(chunks)} chunks "
          f"({sum(c['metadata']['type'] == 'review' for c in chunks)} reviews, "
          f"{sum(c['metadata']['type'] == 'summary' for c in chunks)} summaries).\n")
    print("=== 5 sample chunks ===")
    for c in chunks[:5]:
        print(f"\n--- [{c['metadata']['source']}] type={c['metadata']['type']} "
              f"course={c['metadata']['course']} len={len(c['text'])} ---")
        print(c["text"])
