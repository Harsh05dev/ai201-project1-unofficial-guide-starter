"""Grounded response generation with Groq (llama-3.3-70b-versatile).

Grounding is enforced two ways:
  1. The system prompt instructs the model to answer ONLY from the supplied reviews
     and to decline ("I don't have enough information on that.") when they don't cover
     the question.
  2. Source attribution is appended programmatically from the retrieved chunks'
     metadata, so it never depends on the model choosing to cite.
"""
from groq import Groq

from config import GROQ_API_KEY, LLM_MODEL

SYSTEM_PROMPT = """You are The Unofficial Guide, a board of NJIT students answering \
questions about Computer Science professors using ONLY the student reviews provided \
to you in the context.

Rules:
- Answer using ONLY information found in the provided reviews. Do not use any outside \
or general knowledge about these professors or courses.
- If the provided reviews do not contain enough information to answer, reply exactly: \
"I don't have enough information on that."
- When reviews disagree about a professor, say so and summarize both sides rather than \
picking one.
- Be concise and specific. Quote or paraphrase what students actually said.
- Do not invent ratings, grades, course codes, or professor names that are not in the context."""


def _format_context(chunks):
    """Turn retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[{i}] (source: {c['source']})\n{c['text']}")
    return "\n\n".join(blocks)


def generate_response(query, chunks):
    """Generate a grounded answer. Returns {"answer": str, "sources": [filenames]}."""
    if not chunks:
        return {"answer": "I don't have enough information on that.", "sources": []}

    context = _format_context(chunks)
    user_prompt = (
        f"Student reviews:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the reviews above."
    )

    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # low — keep answers tight to the source text
    )
    answer = completion.choices[0].message.content.strip()

    # Source attribution, appended programmatically (deduped, order preserved).
    sources = list(dict.fromkeys(c["source"] for c in chunks))
    return {"answer": answer, "sources": sources}


def rewrite_followup(history, question):
    """Stretch 4 — rewrite a context-dependent follow-up into a standalone query.

    `history` is a list of (user, assistant) tuples. If there is no history, the
    question is returned unchanged. Otherwise the LLM resolves pronouns/references
    ("is he a tough grader?") into a self-contained query using the recent turns,
    so retrieval has a professor/course to anchor on.
    """
    if not history:
        return question

    recent = history[-3:]
    convo = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in recent)
    prompt = (
        "Given the conversation so far, rewrite the user's latest message into a single "
        "standalone search query about an NJIT CS professor, resolving any pronouns or "
        "references (he/she/they/that class) using the conversation. If the latest message "
        "is already standalone, return it unchanged. Output ONLY the rewritten query, nothing else.\n\n"
        f"Conversation:\n{convo}\n\nLatest message: {question}\n\nStandalone query:"
    )
    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        rewritten = completion.choices[0].message.content.strip().strip('"')
        return rewritten or question
    except Exception:
        return question  # on any failure, fall back to the raw question
