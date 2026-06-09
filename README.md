# The Unofficial Guide — Project 1

A Retrieval-Augmented Generation (RAG) system that answers plain-language questions about
**NJIT Computer Science professors**, grounded in real student reviews with sources cited.

Run it: `python app.py` → builds the vector index on first launch, then opens a Gradio UI at
http://localhost:7860.

---

## Domain

**NJIT Computer Science professor reviews.** The system answers questions about what CS professors
at the New Jersey Institute of Technology are actually like to take — teaching style, exam difficulty,
grading harshness, workload, and whether students recommend them.

This knowledge is valuable and hard to find through official channels because NJIT's course catalog
and faculty pages describe only *what* a course covers, never *how* it's taught. The catalog won't
tell you a class is "bucket-graded" so you can score a 69 and still fail, that a professor "reads off
10-year-old slides," or that "everything on the exam comes from his own handwritten notes." That
signal lives only in student reviews, scattered one professor at a time and impossible to query in
aggregate.

---

## Document Sources

14 documents, one plain-text file per professor, collected from **Rate My Professors** (NJIT,
school id 668). Each file has a metadata header (overall rating, would-take-again %, difficulty,
courses) followed by individual student reviews tagged with course code, date, quality/difficulty
scores, grade, and tags. Corpus: **69 reviews / ~4,800 words**.

| # | Source (professor) | Type | URL or file path |
|---|--------------------|------|-----------------|
| 1 | Ravi Varadarajan | RateMyProfessors review page | https://www.ratemyprofessors.com/professor/2295889 → `documents/ravi_varadarajan.txt` |
| 2 | Shantanu Sharma | RateMyProfessors review page | https://www.ratemyprofessors.com/professor/2724224 → `documents/shantanu_sharma.txt` |
| 3 | Usman Roshan | RateMyProfessors review page | https://www.ratemyprofessors.com/professor/648433 → `documents/usman_roshan.txt` |
| 4 | Cristian Borcea | RateMyProfessors review page | https://www.ratemyprofessors.com/professor/489897 → `documents/cristian_borcea.txt` |
| 5 | Reza Curtmola | RateMyProfessors review page | https://www.ratemyprofessors.com/professor/1168809 → `documents/reza_curtmola.txt` |
| 6 | Marvin Nakayama | RateMyProfessors review page | https://www.ratemyprofessors.com/professor/80111 → `documents/marvin_nakayama.txt` |
| 7 | James Geller | RateMyProfessors review page | https://www.ratemyprofessors.com/professor/81993 → `documents/james_geller.txt` |
| 8 | Ali Mili | RateMyProfessors review page | https://www.ratemyprofessors.com/professor/74719 → `documents/ali_mili.txt` |
| 9 | Vincent Oria | RateMyProfessors review page | https://www.ratemyprofessors.com/professor/386613 → `documents/vincent_oria.txt` |
| 10 | David Nassimi | RateMyProfessors review page | https://www.ratemyprofessors.com/professor/134063 → `documents/david_nassimi.txt` |
| 11 | Andrew Sohn | RateMyProfessors review page | https://www.ratemyprofessors.com/professor/205300 → `documents/andrew_sohn.txt` |
| 12 | Joseph Leung | RateMyProfessors review page | https://www.ratemyprofessors.com/professor/213580 → `documents/joseph_leung.txt` |
| 13 | Chase Wu | RateMyProfessors review page | https://www.ratemyprofessors.com/professor/2251039 → `documents/chase_wu.txt` |
| 14 | Iulian Neamtiu | RateMyProfessors review page | https://www.ratemyprofessors.com/professor/2276510 → `documents/iulian_neamtiu.txt` |

**Ingestion / cleaning process:** Reviews were pulled from the JavaScript-rendered Rate My
Professors pages via an automated fetch, then normalized into a consistent plain-text layout: a
metadata header per professor and one `[Course | date | quality | difficulty | grade | tags]` block
per review followed by the verbatim comment. Navigation, ads, and site boilerplate were dropped;
only the review text and its structured metadata were kept.

---

## Chunking Strategy

**Chunk size:** One student review per chunk (~300–600 characters typical), plus one profile-summary
chunk per professor. Every chunk is prefixed with a `Professor <Name> (<course>)` context header.

**Overlap:** None (0). Review boundaries fall *between* independent opinions, so overlap would bleed
one student's review into another's and contaminate the embedding rather than preserve a fact.

**Why these choices fit the documents:** Reviews are short, self-contained opinions — each is already
an atomic thought. Splitting one mid-sentence strips its meaning; merging several into a fixed-size
block blends 5-star and 1-star opinions into a muddy embedding that matches no specific query. So the
natural boundary *is* the review. Prepending the professor/course header makes even a 3-word review
("Great professor.") standalone-retrievable and attributable.

**Preprocessing before chunking:** split header (profile) from reviews body, split reviews on blank
lines, drop any chunk under 15 characters.

**Final chunk count:** **83** (69 review chunks + 14 profile-summary chunks).

### Sample chunks (5, with source)

```
[1] source: ali_mili.txt  (type=summary)
Professor Ali Mili — profile summary.
Professor: Ali Mili
Department: Computer Science, New Jersey Institute of Technology (NJIT)
Overall rating: 2.1/5 | Would take again: 28% | Difficulty: 3.5/5 | Total ratings: 57
Note: teaches CS610.
```
```
[2] source: andrew_sohn.txt  (type=review, course=CS350)
Professor Andrew Sohn (CS350) — student review.
Course CS350 | Feb 11, 2026 | Quality 4.0/5 | Difficulty 5.0/5 | Grade D | Tags: Tough grader, Respected, Accessible outside class
"He's hard. ... You could have a 69 in the class and still fail since it's bucket-graded. I did well on the first exam and poor on the last exam and had a 67 at first and failed."
```
```
[3] source: james_geller.txt  (type=review, course=CS632)
Professor James Geller (CS632) — student review.
Course CS632 | Dec 23, 2021 | Quality 5.0/5 | Difficulty 2.0/5 | Grade A | Tags: Lots of homework, Beware of pop quizzes, Amazing lectures
"He is one of the best professors in the college. All the lectures are amazing and the class material is also very thorough. His exams are very practical and real-world in nature."
```
```
[4] source: joseph_leung.txt  (type=review, course=CS506)
Professor Joseph Leung (CS506) — student review.
Course CS506 | Dec 22, 2015 | Quality 4.5/5 | Difficulty 4.0/5 | Grade B+
"He gives you a copy of his own notes, and everything on the exam comes from the notes. ... But he doesn't deviate from his notes."
```
```
[5] source: iulian_neamtiu.txt  (type=review, course=CS485)
Professor Iulian Neamtiu (CS485) — student review.
Course CS485 | Dec 3, 2021 | Quality 1.0/5 | Difficulty 3.0/5 | Tags: Lots of homework, Tough grader
"By far one of the worst professors in this school ... Abrasive and completely unhelpful. Gives projects in which he states that he will give no help."
```

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (384-dim, runs locally — no API key,
no rate limits). Chosen because our chunks are short review snippets, exactly the kind of text this
model handles well, and the whole index (83 chunks) embeds in under a second on a laptop.

**Production tradeoff reflection:** Deploying for real users with no cost constraint, I'd weigh:
(a) **domain accuracy** — a larger model (OpenAI `text-embedding-3-large`, Voyage, Cohere v3) embeds
slang and course codes ("bucket-graded," "CS350") more reliably; (b) **context length** — MiniLM caps
near 256 tokens, fine for reviews but limiting if the corpus grows to syllabi/handbooks; (c)
**multilingual** — irrelevant now (English only), relevant if international-student forums were added;
(d) **latency & local vs API** — MiniLM is local and instant, an API model adds network latency and a
per-query bill but offloads compute and upgrades automatically. For this project, local wins on
simplicity and zero cost.

---

## Retrieval Test Results

Embedding model: `all-MiniLM-L6-v2`; top-k = 5; distance = cosine (lower = more similar).

**Query A — "Is Andrew Sohn's CS350 hard, and how is it graded?"**
| distance | professor (course) | source |
|----------|--------------------|--------|
| 0.341 | Andrew Sohn (CS350) | andrew_sohn.txt |
| 0.356 | Andrew Sohn (CS350) | andrew_sohn.txt |
| 0.396 | Andrew Sohn (CS350) | andrew_sohn.txt |
| 0.401 | Andrew Sohn (summary) | andrew_sohn.txt |
| 0.439 | Andrew Sohn (CS350) | andrew_sohn.txt |

*Why relevant:* All 5 chunks come from the correct professor's file and directly address difficulty
and grading — the top hits include the "you could have a 69 and still fail since it's bucket-graded"
review and the profile-summary chunk noting the course is bucket-graded. Tight cluster of low
distances (0.34–0.44) means the query landed squarely on-topic.

**Query B — "What's the main study tip for Joseph Leung's classes?"**
| distance | professor (course) | source |
|----------|--------------------|--------|
| 0.216 | Joseph Leung (CS352) | joseph_leung.txt |
| 0.231 | Joseph Leung (CS332) | joseph_leung.txt |
| 0.231 | Joseph Leung (CS332) | joseph_leung.txt |
| 0.261 | Joseph Leung (CS506) | joseph_leung.txt |
| 0.317 | Joseph Leung (CS332) | joseph_leung.txt |

*Why relevant:* All 5 from `joseph_leung.txt` at very low distances (0.22–0.32). The retrieved
reviews repeatedly stress taking and studying his notes ("everything on the exam comes from the
notes," "good lecture notes are key") — exactly the "study tip" the question asks for.

**Query C — "Which professor do students most recommend for databases?"**
| distance | professor (course) | source |
|----------|--------------------|--------|
| 0.455 | James Geller (summary) | james_geller.txt |
| 0.461 | James Geller (CS632) | james_geller.txt |
| 0.517 | Vincent Oria (summary) | vincent_oria.txt |
| 0.542 | Cristian Borcea (CS643) | cristian_borcea.txt |
| 0.557 | Chase Wu (CS644) | chase_wu.txt |

Higher distances here (0.45+) because "databases" is a broad topical query and several professors
teach data-adjacent courses — but James Geller (CS632 Databases, the highest-rated professor) correctly
surfaces on top, which is the right answer.

---

## Grounded Generation

**System prompt grounding instruction** (see `generator.py`): the model is told to *"Answer using ONLY
information found in the provided reviews. Do not use any outside or general knowledge,"* and *"If the
provided reviews do not contain enough information to answer, reply exactly: 'I don't have enough
information on that.'"* Temperature is set to 0.2 to keep answers tight to the source text. When
reviews disagree, the prompt instructs the model to present both sides rather than pick one.

**How source attribution is surfaced:** Source filenames are appended **programmatically** from the
retrieved chunks' metadata (in `app.py` / `generator.py`), not left to the model to volunteer — so
every answered query shows which professor files it drew from, with distance scores. On a refusal the
UI shows "(no relevant reviews found)" instead of misleading sources.

### Example responses (with source attribution)

**Example 1 — "Is Ali Mili an easy professor?"**
> No, according to the reviews, Professor Ali Mili is not an easy professor. Multiple reviews mention
> that his exams are "very tough" and the class is "hard as hell". One review states that the course is
> "Test heavy" and another mentions that the professor is a "Tough grader". The difficulty ratings in
> the reviews range from 3.0/5 to 5.0/5, indicating a challenging course.
>
> **Retrieved from:** ali_mili.txt — Ali Mili (CS610) ×5 [distances 0.279–0.411]

**Example 2 — "What's the main study tip for Joseph Leung's classes?"**
> According to the reviews, the main study tip for Joseph Leung's classes is to take good notes — one
> student said "make sure to take all the notes" and another stated "Good lecture notes are key."
> Reviewing the lecture notes and matching them with the textbook and homework is also recommended.
> One student also advised to "Sit in the front row" to deal with the professor's small handwriting.
>
> **Retrieved from:** joseph_leung.txt — Joseph Leung (CS332/CS352/CS506) ×5 [distances 0.211–0.314]

**Out-of-scope query — "What's the best dorm at NJIT?"**
> I don't have enough information on that.
>
> **Retrieved from:** (no relevant reviews found) — nearest chunks were unrelated CS reviews at
> distances 0.567+, and the grounding instruction made the model decline rather than invent a dorm
> answer from general knowledge.

---

## Evaluation Report

Run with `python eval.py` (retrieve → generate over all 5 questions + 1 out-of-scope test).

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | Is Andrew Sohn's CS350 hard, and how is it graded? | Hard (4.1 difficulty); bucket-graded — a 69 can still fail; reviews split on teaching | "Hard per some students; difficulty ratings split 3.0–5.0; bucket-graded, so a 67% can still fail." Cited 5 Sohn chunks. | Relevant (5/5 from Sohn) | **Accurate** |
| 2 | Which professor do students most recommend for databases? | James Geller — 4.9/5, 100% would take again, CS632 | "James Geller — 4.9/5, 100% would take again, CS632 db admin praise." | Relevant (Geller top) | **Accurate** |
| 3 | Do students recommend Cristian Borcea for CS643 Cloud Computing? | Mostly no for CS643 (0% would take again, outdated slides, theory-only) | "Mixed: cites 'one of the worst' vs 'great professor'; notes 0% would take again." Presented both sides. | Partially relevant (2/5 chunks were CS656 networking, not CS643 cloud) | **Partially accurate** |
| 4 | What's the main study tip for Joseph Leung's classes? | Rely on his own handwritten notes; exams come from the notes | "Take good notes; 'good lecture notes are key'; sit in front row." | Relevant (5/5 from Leung) | **Accurate** |
| 5 | Is Ali Mili an easy professor? | No — 2.1/5, tough exams, reads off slides; one review calls him approachable | "No — exams 'very tough', 'hard as hell', test heavy, tough grader." | Relevant (5/5 from Mili) | **Accurate** |

**Out-of-scope refusal test:** "What's the best dorm at NJIT?" → system replied *"I don't have enough
information on that."* Nearest chunks were unrelated CS reviews at distances 0.567+. **Correct refusal.**

---

## Failure Case Analysis

**Question that failed:** "Do students recommend Cristian Borcea for CS643 Cloud Computing?"

**What the system returned:** A reasonable, balanced answer ("mixed — some call him one of the worst,
others say great professor; 0% would take again"). But of the 5 chunks retrieved, **2 came from his
CS656 Networking reviews, not CS643 Cloud Computing** — distances 0.363 and 0.391, ranked above one of
the actual CS643 reviews. So 40% of the context the model reasoned over was about the *wrong course*.

**Root cause (tied to a specific pipeline stage):** The **embedding/retrieval** stage. Semantic
similarity ranks all of a professor's reviews close together because the dominant signal is "this is a
review of Cristian Borcea," while the short course-code token ("CS643" vs "CS656") carries weak
embedding weight. The chunker prepends the professor name to every chunk, which *helps* attribution but
*reinforces* this same-professor clustering. The query named a specific course, but cosine similarity
had no reason to prefer CS643 chunks over CS656 chunks from the same professor.

**What you would change to fix it:** Add **metadata filtering**: when the query names a course (regex
for `CS\d+`), pass a ChromaDB `where={"course": "CS643"}` filter so retrieval is constrained to that
course's chunks (and only falls back to all-professor chunks if too few match). This is a listed
stretch feature (metadata filtering) and directly targets the root cause without changing the
embedding model.

---

## Spec Reflection

**One way the spec helped you during implementation:** Deciding the chunking strategy in `planning.md`
*before* coding — one review per chunk with a prepended professor/course header — meant `ingest.py`
had an exact target. The predicted chunk count (~83) matched the actual output (83) on the first run,
and retrieval landed on-target immediately because each chunk was self-contained and attributable by
design rather than by trial and error.

**One way your implementation diverged from the spec, and why:** The spec didn't mention what to do
with source attribution when the system *declines* to answer. During the out-of-scope test ("best
dorm"), retrieval still returns its 5 nearest chunks (CS professor reviews), so naively listing them as
"sources" would falsely imply those reviews backed a dorm answer. I diverged by having `app.py` detect
the refusal string and show "(no relevant reviews found)" instead of the chunk list. I also added a
profile-summary chunk per professor (not just per-review chunks) so questions about ratings and
would-take-again percentages have something to match — both changes came from testing, not the
original plan.

---

## AI Usage

**Instance 1 — Document collection**
- *What I gave the AI:* The domain (NJIT CS professor reviews) and a request to gather 10+ source
  documents from Rate My Professors.
- *What it produced:* It searched for NJIT CS professors' RMP profile IDs and fetched each page,
  extracting per-review text, scores, course codes, dates, and tags into 14 structured `.txt` files.
- *What I changed or overrode:* I directed the file format (metadata header + one block per review)
  so the documents would chunk cleanly, chose which 14 professors to include for a spread of ratings
  (top-rated, low-rated, polarizing), and kept the structured per-review metadata (course, date,
  scores, tags) rather than just the comment text so it could be used in chunks and attribution.

**Instance 2 — Pipeline implementation**
- *What I gave the AI:* My `planning.md` Chunking Strategy and Retrieval Approach sections plus the
  architecture diagram.
- *What it produced:* `ingest.py` (review-boundary chunker with prepended headers), `retriever.py`
  (MiniLM + ChromaDB, top-k=5), and `generator.py` (grounded Groq prompt with programmatic citations).
- *What I changed or overrode:* I kept top-k at 5 (verified it gathered multiple opinions on
  polarizing professors without dragging in unrelated ones), set generation temperature to 0.2 for
  tighter grounding, and required source attribution to be appended **programmatically** from chunk
  metadata rather than trusting the model to cite — so a citation can never be hallucinated or omitted.

---

## Query Interface

A **Gradio web UI** (`app.py`), launched with `python app.py` at http://localhost:7860.

**Input field:** a single textbox — "Your question" — where the user types a plain-language question
about an NJIT CS professor. Six clickable example questions are provided. Pressing Enter or the **Ask**
button submits.

**Output fields:**
- **Answer** — the grounded, generated response.
- **Retrieved from** — the source files behind the answer, each line showing
  `filename — Professor (course) [distance]`, so the user can see exactly which reviews informed it
  (or "(no relevant reviews found)" on a refusal).

**Sample interaction transcript:**
```
Your question:  Is Andrew Sohn's CS350 hard, and how is it graded?

Answer:         Andrew Sohn's CS350 is considered hard by some students — difficulty ratings
                in the reviews split between 3.0 and 5.0 out of 5. The class is "bucket-graded"
                across multiple reviews, meaning grades are grouped into ranges and a student
                can fail even with a high percentage (one student had a 67% and still failed).

Retrieved from: • andrew_sohn.txt — Andrew Sohn (CS350)  [distance 0.341]
                • andrew_sohn.txt — Andrew Sohn (CS350)  [distance 0.356]
                • andrew_sohn.txt — Andrew Sohn (CS350)  [distance 0.396]
                • andrew_sohn.txt — Andrew Sohn (N/A)     [distance 0.401]
                • andrew_sohn.txt — Andrew Sohn (CS350)  [distance 0.439]
```
