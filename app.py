"""The Unofficial Guide — NJIT CS professor reviews RAG app.

Run:  python app.py   → builds the index on first launch, then opens the Gradio UI
at http://localhost:7860.

Features:
  - Grounded answers from real Rate My Professors reviews, with sources + distances
  - Stretch 1: metadata filters (course dropdown, minimum-rating slider) + auto
    course detection from the question
  - Stretch 2: search mode toggle — Semantic or Hybrid (semantic + BM25)
  - Stretch 4: conversational memory — multi-turn follow-ups are rewritten into
    standalone queries using the chat history
"""
import re

import gradio as gr

from config import N_RESULTS
from retriever import ensure_index, retrieve, list_courses
from hybrid import hybrid_retrieve
from generator import generate_response, rewrite_followup

REFUSAL = "I don't have enough information on that."
COURSE_RE = re.compile(r"\b(?:CS|CIS|MIS)\s?\d{3,4}\w*\b", re.IGNORECASE)


def _detect_course(text):
    m = COURSE_RE.search(text)
    return m.group(0).upper().replace(" ", "") if m else None


def _history_pairs(messages):
    """Turn Gradio 'messages' history into (user, assistant) pairs for the rewriter."""
    pairs, last_user = [], None
    for m in messages:
        if m["role"] == "user":
            last_user = m["content"]
        elif m["role"] == "assistant" and last_user is not None:
            pairs.append((last_user, m["content"]))
            last_user = None
    return pairs


def _format_sources(chunks):
    seen, lines = set(), []
    for r in chunks:
        key = (r["source"], r["course"])
        if key in seen:
            continue
        seen.add(key)
        dist = "" if r["distance"] is None else f" [dist {r['distance']:.3f}]"
        lines.append(f"• {r['source']} — {r['professor']} ({r['course']}){dist}")
    return "\n".join(lines)


def respond(message, messages, mode, course_filter, min_rating):
    if not message or not message.strip():
        return messages, ""

    # Stretch 4 — rewrite follow-ups into a standalone query using history.
    standalone = rewrite_followup(_history_pairs(messages), message)

    # Stretch 1 — filters: explicit course dropdown wins; else auto-detect from query.
    course = course_filter or _detect_course(standalone)
    min_r = float(min_rating) if min_rating and float(min_rating) > 0 else None

    # Stretch 2 — search mode.
    if mode.startswith("Hybrid"):
        chunks = hybrid_retrieve(standalone, k=N_RESULTS)
        if course:  # hybrid has no native filter — apply course post-hoc, keep all if none match
            chunks = [c for c in chunks if c["course"] == course] or chunks
    else:
        chunks = retrieve(standalone, k=N_RESULTS, course=course, min_rating=min_r)

    result = generate_response(standalone, chunks)
    answer = result["answer"]

    if answer.strip().startswith(REFUSAL):
        body = f"{answer}\n\n_(no relevant reviews found)_"
    else:
        body = f"{answer}\n\n**Retrieved from:**\n{_format_sources(chunks)}"

    if standalone.strip().lower() != message.strip().lower():
        body += f"\n\n_↳ interpreted your question as: “{standalone}”_"

    messages = messages + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": body},
    ]
    return messages, ""


def build_ui():
    courses = [""] + list_courses()
    with gr.Blocks(title="The Unofficial Guide — NJIT CS Professors") as demo:
        gr.Markdown(
            "# 🎓 The Unofficial Guide — NJIT CS Professors\n"
            "Ask what students *really* say about NJIT Computer Science professors. "
            "Answers are grounded in real Rate My Professors reviews, with sources shown. "
            "Supports follow-up questions."
        )
        with gr.Row():
            mode = gr.Radio(
                ["Semantic", "Hybrid (semantic + BM25)"],
                value="Semantic", label="Search mode", scale=2,
            )
            course_filter = gr.Dropdown(
                courses, value="", label="Filter by course (optional)", scale=1,
            )
            min_rating = gr.Slider(
                0, 5, value=0, step=0.1, label="Min professor rating (optional)", scale=1,
            )

        chatbot = gr.Chatbot(height=420, label="Conversation")
        msg = gr.Textbox(
            label="Your question",
            placeholder="e.g. Is Andrew Sohn's CS350 hard?  …then: is he a tough grader?",
        )
        with gr.Row():
            send = gr.Button("Ask", variant="primary")
            clear = gr.Button("Clear chat")

        gr.Examples(
            examples=[
                "Is Andrew Sohn's CS350 hard, and how is it graded?",
                "Which professor do students most recommend for databases?",
                "Do students recommend Cristian Borcea for CS643 Cloud Computing?",
                "What's the main study tip for Joseph Leung's classes?",
                "What's the best dorm at NJIT?",
            ],
            inputs=msg,
        )

        send.click(respond, [msg, chatbot, mode, course_filter, min_rating], [chatbot, msg])
        msg.submit(respond, [msg, chatbot, mode, course_filter, min_rating], [chatbot, msg])
        clear.click(lambda: ([], ""), outputs=[chatbot, msg])
    return demo


if __name__ == "__main__":
    print("Building / loading vector index...")
    n = ensure_index()
    print(f"Index ready with {n} chunks. Launching UI...")
    build_ui().launch()
