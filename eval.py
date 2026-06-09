"""Evaluation runner (Milestone 6).

Runs the 5 test questions from planning.md (plus one out-of-scope refusal test)
through the full retrieve → generate pipeline and prints each answer with its
retrieved sources and distances. Used to fill the Evaluation Report in README.md.
"""
from retriever import ensure_index, retrieve
from generator import generate_response
from config import N_RESULTS

QUESTIONS = [
    "Is Andrew Sohn's CS350 hard, and how is it graded?",
    "Which professor do students most recommend for databases?",
    "Do students recommend Cristian Borcea for CS643 Cloud Computing?",
    "What's the main study tip for Joseph Leung's classes?",
    "Is Ali Mili an easy professor?",
    "What's the best dorm at NJIT?",  # out-of-scope refusal test
]


def main():
    ensure_index()
    for i, q in enumerate(QUESTIONS, 1):
        chunks = retrieve(q, k=N_RESULTS)
        result = generate_response(q, chunks)
        print("=" * 80)
        print(f"Q{i}: {q}")
        print("-" * 80)
        print(f"ANSWER:\n{result['answer']}\n")
        print("RETRIEVED CHUNKS:")
        for r in chunks:
            print(f"  [{r['distance']:.3f}] {r['professor']} ({r['course']}) <{r['source']}>")
        print()


if __name__ == "__main__":
    main()
