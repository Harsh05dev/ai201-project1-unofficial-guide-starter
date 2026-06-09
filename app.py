"""The Unofficial Guide — NJIT CS professor reviews RAG app.

Run:  python app.py   → builds the index on first launch, then opens the Gradio UI
at http://localhost:7860.
"""
import gradio as gr

from config import N_RESULTS
from retriever import ensure_index, retrieve
from generator import generate_response

REFUSAL = "I don't have enough information on that."


def ask(question):
    """End-to-end: retrieve relevant review chunks, then generate a grounded answer."""
    chunks = retrieve(question, k=N_RESULTS)
    result = generate_response(question, chunks)
    return result, chunks


def handle_query(question):
    """Gradio handler → (answer_text, sources_text)."""
    if not question or not question.strip():
        return "Ask a question about an NJIT CS professor.", ""

    result, chunks = ask(question)
    answer = result["answer"]

    # If the system declined, don't attach misleading sources.
    if answer.strip().startswith(REFUSAL):
        return answer, "(no relevant reviews found)"

    lines = []
    for r in chunks:
        lines.append(f"• {r['source']} — {r['professor']} ({r['course']})  [distance {r['distance']:.3f}]")
    sources_text = "\n".join(lines)
    return answer, sources_text


def build_ui():
    with gr.Blocks(title="The Unofficial Guide — NJIT CS Professors") as demo:
        gr.Markdown(
            "# 🎓 The Unofficial Guide — NJIT CS Professors\n"
            "Ask what students *really* say about NJIT Computer Science professors. "
            "Answers are grounded in real Rate My Professors reviews, with sources shown."
        )
        inp = gr.Textbox(
            label="Your question",
            placeholder="e.g. Is Andrew Sohn's CS350 hard, and how is it graded?",
        )
        btn = gr.Button("Ask", variant="primary")
        answer = gr.Textbox(label="Answer", lines=8)
        sources = gr.Textbox(label="Retrieved from", lines=6)

        gr.Examples(
            examples=[
                "Is Andrew Sohn's CS350 hard, and how is it graded?",
                "Which professor do students most recommend for databases?",
                "Do students recommend Cristian Borcea for CS643 Cloud Computing?",
                "What's the main study tip for Joseph Leung's classes?",
                "Is Ali Mili an easy professor?",
                "What's the best dorm at NJIT?",
            ],
            inputs=inp,
        )

        btn.click(handle_query, inputs=inp, outputs=[answer, sources])
        inp.submit(handle_query, inputs=inp, outputs=[answer, sources])
    return demo


if __name__ == "__main__":
    print("Building / loading vector index...")
    n = ensure_index()
    print(f"Index ready with {n} chunks. Launching UI...")
    build_ui().launch()
