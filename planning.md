# Project 1 Planning: The Unofficial Guide

> Written before building the pipeline. Technical sections (chunking, retrieval) are
> updated if the approach changes during implementation.

---

## Domain

**NJIT Computer Science professor reviews.** The system answers plain-language questions
about what CS professors at the New Jersey Institute of Technology are actually like to take —
teaching style, exam difficulty, grading harshness, workload, and whether students recommend them.

This knowledge is valuable and hard to find through official channels because NJIT's course
catalog and faculty pages only describe *what* a course covers, never *how* it is taught. The
catalog won't tell you that a class is "bucket-graded" so you can score a 69 and still fail, that
a professor "reads off 10-year-old slides," or that "everything on the exam comes from his own
handwritten notes." That signal lives only in student-written reviews, scattered one professor at
a time across Rate My Professors, and impossible to query in aggregate ("which professor is the
best option for CS350?").

---

## Documents

14 documents, one plain-text file per professor, collected from **Rate My Professors**
(NJIT, school id 668). Each file has a metadata header (overall rating, would-take-again %,
difficulty, courses taught) followed by individual student reviews. Each review block records the
course code, date, quality score, difficulty score, grade, and tags, plus the verbatim comment.
Corpus totals **69 student reviews / ~4,800 words**.

| #  | Source (professor) | Description | URL or location |
|----|--------------------|-------------|-----------------|
| 1  | Ravi Varadarajan | CS114 instructor, low-rated, polarizing | https://www.ratemyprofessors.com/professor/2295889 → `documents/ravi_varadarajan.txt` |
| 2  | Shantanu Sharma | CS331, mostly positive, "caring" | https://www.ratemyprofessors.com/professor/2724224 → `documents/shantanu_sharma.txt` |
| 3  | Usman Roshan | CS675/CS677 ML, tough grader | https://www.ratemyprofessors.com/professor/648433 → `documents/usman_roshan.txt` |
| 4  | Cristian Borcea | CS643 Cloud / CS656 Networking, mixed | https://www.ratemyprofessors.com/professor/489897 → `documents/cristian_borcea.txt` |
| 5  | Reza Curtmola | CS645 security, mixed, tough grader | https://www.ratemyprofessors.com/professor/1168809 → `documents/reza_curtmola.txt` |
| 6  | Marvin Nakayama | CS341, highly rated, lots of resources | https://www.ratemyprofessors.com/professor/80111 → `documents/marvin_nakayama.txt` |
| 7  | James Geller | CS632 Databases, top-rated (4.9) | https://www.ratemyprofessors.com/professor/81993 → `documents/james_geller.txt` |
| 8  | Ali Mili | CS610, low-rated, tough exams | https://www.ratemyprofessors.com/professor/74719 → `documents/ali_mili.txt` |
| 9  | Vincent Oria | CS631 Data Mgmt, grading complaints | https://www.ratemyprofessors.com/professor/386613 → `documents/vincent_oria.txt` |
| 10 | David Nassimi | CS435 Algorithms, polarizing (deceased) | https://www.ratemyprofessors.com/professor/134063 → `documents/david_nassimi.txt` |
| 11 | Andrew Sohn | CS350 Systems, hard, bucket-graded | https://www.ratemyprofessors.com/professor/205300 → `documents/andrew_sohn.txt` |
| 12 | Joseph Leung | CS332 OS / CS506, "study his notes" | https://www.ratemyprofessors.com/professor/213580 → `documents/joseph_leung.txt` |
| 13 | Chase Wu | CS644 Big Data/Hadoop, mixed | https://www.ratemyprofessors.com/professor/2251039 → `documents/chase_wu.txt` |
| 14 | Iulian Neamtiu | CS388/CS485/CS673, "shouty tone" | https://www.ratemyprofessors.com/professor/2276510 → `documents/iulian_neamtiu.txt` |

Chosen for variety: high-rated (Geller 4.9, Nakayama 4.3), low-rated (Varadarajan 1.7, Mili/Neamtiu 2.1),
and polarizing (same professor with both 1.0 and 5.0 reviews), spanning core undergrad courses
(CS114, CS350, CS435) and graduate electives (CS610, CS632, CS643, CS644, CS645).

---

## Chunking Strategy

**Chunk size:** One student review per chunk (typically ~300–600 characters), plus one
"profile summary" chunk per professor (the header lines: rating, would-take-again, difficulty,
courses). Every chunk is prefixed with a context header — `Professor <Name> (<course>):` — so
each chunk carries who and what it is about.

**Overlap:** None (0 characters) between reviews.

**Reasoning:** Reviews are short, self-contained opinions — each one is already an atomic "thought"
(see the good-chunk example in the milestone). Splitting a review mid-sentence would strip its meaning;
merging several reviews into one fixed-size chunk would blend a 5-star and a 1-star opinion into a
muddy embedding that matches no specific query well. So the natural boundary *is* the review.
Overlap exists to keep a fact from being severed at a boundary, but here boundaries fall *between*
independent reviews — bleeding one review's text into the next would contaminate, not help, so
overlap is 0. The risk with per-review chunks is that very short reviews ("Great professor.") have
no standalone meaning; prepending the `Professor <Name> (<course>):` header fixes that — it becomes
"Professor James Geller (CS632): Great professor." which is retrievable and attributable on its own.

**Preprocessing before chunking:** strip the source URL/metadata noise into a dedicated summary
chunk, drop empty lines, filter out any chunk under ~15 characters.

**Expected final chunk count:** ~83 (69 review chunks + 14 profile summary chunks). Comfortably
inside the 50–2,000 sane-range from the milestone guidance.

---

## Retrieval Approach

**Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers` — runs locally, no API key,
no rate limits, 384-dim, strong on short-text semantic similarity. Good fit because our chunks are
short review snippets, exactly the kind of text this model was trained on.

**Top-k:** 5. With ~83 small chunks and questions that usually target one professor, 5 chunks is
enough to gather multiple opinions on the same professor (so the answer reflects consensus, not one
outlier) without dragging in unrelated professors. Too few (k=1–2) risks missing the balancing
opinion on a polarizing professor; too many (k=10+) pulls in other professors' reviews and pulls the
LLM off-target.

**Why semantic search works here:** a query like "is this class hard to pass" matches a review
saying "you could have a 69 and still fail" even with zero shared keywords, because the embedding
captures meaning, not surface words.

**Production tradeoff reflection:** If deploying for real users with no cost constraint, I'd weigh:
(a) **domain accuracy** — a larger model (OpenAI `text-embedding-3-large`, Voyage, Cohere v3) embeds
slang and course codes ("bucket-graded," "CS350") more reliably; (b) **context length** — MiniLM
truncates at 256 tokens, fine for reviews but limiting for long-form guides, so a longer-context model
matters if the corpus expands to syllabi/handbooks; (c) **multilingual** — irrelevant now (English-only),
but relevant if international-student forums were added; (d) **latency & local vs API** — MiniLM is
local and instant; an API model adds network latency and a per-query bill but offloads compute and
upgrades automatically. For this project the local model wins on simplicity and zero cost.

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | "Is Andrew Sohn's CS350 hard, and how is it graded?" | Yes, hard (difficulty 4.1). It is **bucket-graded**, so you can have a 69 and still fail. Reviews are split — some call his teaching unclear and say avoid; others finished with an A and recommend him. |
| 2 | "Which professor do students most recommend for databases / CS632?" | **James Geller** — 4.9/5, 100% would take again, "amazing lectures," "one of the best professors in the college." |
| 3 | "Do students recommend taking Cristian Borcea for CS643 Cloud Computing?" | Mostly **no** for recent CS643 — 0% would take again, complaints about 10-year-old slides and theory-only teaching. (His older CS656 networking reviews are positive — a chance for the system to overstate the negative.) |
| 4 | "What's the main study tip for Joseph Leung's classes?" | **Rely on his own handwritten notes** — "everything on the exam comes from the notes"; lectures don't match the textbook. |
| 5 | "Is Ali Mili an easy professor?" | **No** — 2.1/5, only 28% would take again, exams "very tough," "reads off the slides." (One review still calls him approachable, so a fair answer is "no, but approachable.") |

**Out-of-scope refusal test:** "What's the best dorm at NJIT?" → system must decline (no housing data in corpus), not invent an answer.

---

## Anticipated Challenges

1. **Polarizing professors → one-sided answers.** Several professors have both 1.0 and 5.0 reviews
   (Sohn, Borcea, Curtmola). If top-k retrieval happens to pull only the negative (or only the
   positive) reviews, the LLM will give a confidently lopsided answer that misrepresents the consensus.
   Mitigation: k=5 to gather multiple opinions; prompt the model to note disagreement.

2. **Course-code queries can mis-route.** Short tokens like "CS350" or "CS643" carry weak semantic
   signal, so a query naming a course could retrieve the right *topic* from the wrong *professor*
   (e.g., another database course). Mitigation: prepend `Professor <Name> (<course>):` to every chunk
   so the course code is embedded in context, and surface source filenames so mis-routing is visible.

3. **Thin coverage per professor.** Some files hold only 4–5 reviews. A generic query ("who is a good
   professor?") with no professor named has little to anchor on and may return a scattered mix.

4. **Extraction noise.** Reviews were pulled from a JavaScript-rendered site via an automated fetch
   that occasionally lightly paraphrased a comment. Minor wording drift could slightly weaken grounding
   fidelity; the metadata (scores, tags, dates) is reliable.

---

## Architecture

```
┌──────────────────┐   ┌──────────────────┐   ┌─────────────────────────┐
│ 1. INGESTION     │   │ 2. CHUNKING      │   │ 3. EMBED + VECTOR STORE │
│ load 14 .txt     │──▶│ split per review │──▶│ all-MiniLM-L6-v2        │
│ from documents/  │   │ + prof/course    │   │ embeddings →            │
│ (Python stdlib)  │   │ header, k=0 ovlp │   │ ChromaDB (+ metadata:   │
│                  │   │                  │   │ source file, prof,      │
│                  │   │                  │   │ course, chunk #)        │
└──────────────────┘   └──────────────────┘   └────────────┬────────────┘
                                                            │
                                user query                  ▼
                          ┌───────────────────────────────────────────┐
                          │ 4. RETRIEVAL                              │
                          │ embed query → ChromaDB similarity search  │
                          │ → top-k=5 chunks + source metadata        │
                          └───────────────────┬───────────────────────┘
                                              ▼
                          ┌───────────────────────────────────────────┐
                          │ 5. GENERATION                             │
                          │ Groq llama-3.3-70b-versatile              │
                          │ prompt = grounding instruction + chunks   │
                          │ → answer + source attribution             │
                          │ surfaced in Gradio UI                     │
                          └───────────────────────────────────────────┘
```

Stage tools: ingestion = Python `os`/`pathlib`; chunking = custom splitter on review boundaries;
embedding = `sentence-transformers` (`all-MiniLM-L6-v2`); vector store = `chromadb`;
generation = `groq` (`llama-3.3-70b-versatile`); interface = `gradio`.

---

## AI Tool Plan

Tool used throughout: **Claude (via Claude Code)**.

**Milestone 3 — Ingestion and chunking:**
Input I'll give Claude: this Documents section (the per-professor `.txt` format with a metadata
header and `[Course ... ]` review blocks) and this Chunking Strategy section (one review per chunk,
prof/course header prepended, 0 overlap, profile-summary chunk per professor). I expect it to produce
`ingest.py` with a `load_documents()` that reads `documents/*.txt` and a `chunk_document()` that splits
on the `[Course` review markers, prepends the header, emits a summary chunk, and attaches metadata
(source filename, professor, course). Verification: print 5 chunks and confirm each is self-contained
and that the total chunk count lands near the predicted ~83.

**Milestone 4 — Embedding and retrieval:**
Input: this Retrieval Approach section + the architecture diagram. I expect `retriever.py` with
`embed_and_store()` (embed chunks with `all-MiniLM-L6-v2`, store in ChromaDB with metadata) and
`retrieve(query, k=5)` returning chunks + sources + distances. Verification: run eval questions 1, 2, 4
and confirm top results come from the correct professor's file with distances below ~0.5.

**Milestone 5 — Generation and interface:**
Input: the grounding requirement (answer only from retrieved chunks, decline when unsupported),
desired output format (answer + cited source files), and the Gradio skeleton. I expect `generator.py`
with a `generate_response()` whose system prompt *enforces* grounding, plus `app.py` wiring retrieval →
generation → UI with source attribution appended programmatically (not left to the LLM). Verification:
run the out-of-scope dorm question and confirm the system refuses rather than hallucinating.

---

## Stretch Features

> Planned (per assignment) before implementation. All four stretch features are implemented.

### Stretch 1 — Metadata filtering
**Plan:** Let the user (or the query itself) constrain retrieval by `professor`, `course`, or minimum
overall `rating`. ChromaDB supports a `where=` filter on stored metadata. I'll add a `filters` argument
to `retrieve()` and, in the UI, a course dropdown + a "minimum rating" slider. This *directly fixes the
documented failure case* (the CS643 query that pulled CS656 chunks): when a course is selected,
retrieval is limited to that course's chunks. **Why it fits:** our chunks already carry `professor`,
`course`, and (added for this) a numeric `rating` field, so filtering is a metadata `where` clause, not
a re-embed.

### Stretch 2 — Hybrid search (semantic + BM25)
**Plan:** Add lexical BM25 (via `rank-bm25`) over the same chunks and fuse it with the existing
semantic scores using Reciprocal Rank Fusion (RRF). Compare semantic-only vs hybrid on the eval query
set. **Why it fits:** course codes ("CS643") and proper nouns ("Hadoop", professor names) are exact
tokens that lexical search nails but dense embeddings under-weight — the same weakness behind the
failure case. Hybrid should recover course/keyword precision while keeping semantic recall.

### Stretch 3 — Chunking strategy comparison
**Plan:** Implement a second chunker — fixed ~500-character windows with 100-char overlap (ignoring
review boundaries) — and run both chunkers through the same embed → retrieve pipeline on the eval
queries, reporting average top-1 distance and whether the right professor was retrieved. **Hypothesis:**
the per-review chunker wins because it respects opinion boundaries; the fixed-size chunker will merge a
5-star and 1-star review into one window and blur the embedding. A standalone `compare_chunking.py`
prints the comparison; results go in the README.

### Stretch 4 — Conversational memory
**Plan:** Support multi-turn follow-ups ("is he a tough grader?" after asking about a professor) by
keeping the chat history and, before retrieval, rewriting a context-dependent follow-up into a
standalone query using the LLM + recent turns. The Gradio UI becomes a `gr.Chatbot`. **Why it fits:**
students ask follow-ups naturally; without rewriting, "is he hard?" has no professor to anchor
retrieval, so the rewrite step resolves the pronoun from history before searching.
