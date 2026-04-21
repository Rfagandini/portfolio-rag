"""
Evaluate the RAG pipeline against 100 test questions.
Uses LLM-as-judge to auto-grade answers as CORRECT / PARTIAL / INCORRECT.
Reports overall accuracy + breakdown by document.
"""

import json
import time
import os

from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()

from rag_chain import build_chain, get_session_history
from test_questions import test_questions, follow_up_pairs


# --- GRADER SETUP ---

GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict grader. Compare the RAG answer to the expected answer. "
     "Score as CORRECT if the key fact from the expected answer is clearly present "
     "in the RAG answer. Score as PARTIAL if the RAG answer contains some but not all "
     "key facts. Score as INCORRECT if the key fact is missing, wrong, or the RAG says "
     "it cannot answer. Reply with ONLY one word: CORRECT, PARTIAL, or INCORRECT."),
    ("human", "Expected answer: {expected}\n\nRAG answer: {rag_answer}")
])


def create_grader():
    llm = ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        openai_api_key=os.environ.get("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0,
    )
    return GRADER_PROMPT | llm | StrOutputParser()


def grade_answer(grader, expected: str, rag_answer: str) -> str:
    result = grader.invoke({"expected": expected, "rag_answer": rag_answer})
    result = result.strip().upper()
    for valid in ["CORRECT", "PARTIAL", "INCORRECT"]:
        if valid in result:
            return valid
    return "INCORRECT"


# --- EVALUATION ---

def get_follow_up_ids():
    """Return set of question IDs that are follow-ups (second in each pair)."""
    return {pair[1] for pair in follow_up_pairs}


def get_question_by_id(qid: int):
    for q in test_questions:
        if q["id"] == qid:
            return q
    return None


def evaluate_standalone(chain, grader):
    """Evaluate all non-follow-up questions, each with a fresh session."""
    follow_up_ids = get_follow_up_ids()
    # Also skip the first question in each follow-up pair (evaluated in evaluate_follow_ups)
    follow_up_first_ids = {pair[0] for pair in follow_up_pairs}
    skip_ids = follow_up_ids | follow_up_first_ids

    results = []
    standalone_qs = [q for q in test_questions if q["id"] not in skip_ids]

    for i, q in enumerate(standalone_qs):
        config = {"configurable": {"session_id": f"eval-standalone-{q['id']}"}}
        try:
            response = chain.invoke({"input": q["input"]}, config=config)
            rag_answer = response["answer"]
        except Exception as e:
            rag_answer = f"ERROR: {e}"

        grade = grade_answer(grader, q["expected"], rag_answer)

        results.append({
            "id": q["id"],
            "question": q["input"],
            "expected": q["expected"],
            "rag_answer": rag_answer,
            "grade": grade,
            "is_follow_up": False
        })

        print(f"  [{i+1}/{len(standalone_qs)}] Q{q['id']}: {grade}")
        time.sleep(1.5)  # Groq rate limit

    return results


def evaluate_follow_ups(chain, grader):
    """Evaluate follow-up pairs: first question + follow-up in same session."""
    results = []

    for i, (first_id, follow_id) in enumerate(follow_up_pairs):
        q1 = get_question_by_id(first_id)
        q2 = get_question_by_id(follow_id)
        config = {"configurable": {"session_id": f"eval-followup-{first_id}-{follow_id}"}}

        # First question
        try:
            r1 = chain.invoke({"input": q1["input"]}, config=config)
            answer1 = r1["answer"]
        except Exception as e:
            answer1 = f"ERROR: {e}"

        grade1 = grade_answer(grader, q1["expected"], answer1)
        results.append({
            "id": q1["id"],
            "question": q1["input"],
            "expected": q1["expected"],
            "rag_answer": answer1,
            "grade": grade1,
            "is_follow_up": False
        })

        print(f"  [Pair {i+1}/{len(follow_up_pairs)}] Q{q1['id']}: {grade1}")
        time.sleep(1.5)

        # Follow-up (same session, history should help)
        try:
            r2 = chain.invoke({"input": q2["input"]}, config=config)
            answer2 = r2["answer"]
        except Exception as e:
            answer2 = f"ERROR: {e}"

        grade2 = grade_answer(grader, q2["expected"], answer2)
        results.append({
            "id": q2["id"],
            "question": q2["input"],
            "expected": q2["expected"],
            "rag_answer": answer2,
            "grade": grade2,
            "is_follow_up": True
        })

        print(f"  [Pair {i+1}/{len(follow_up_pairs)}] Q{q2['id']} (follow-up): {grade2}")
        time.sleep(1.5)

    return results


# --- REPORTING ---

def print_report(results):
    correct = sum(1 for r in results if r["grade"] == "CORRECT")
    partial = sum(1 for r in results if r["grade"] == "PARTIAL")
    incorrect = sum(1 for r in results if r["grade"] == "INCORRECT")
    total = len(results)

    score = (correct + 0.5 * partial) / total * 100

    print("\n" + "=" * 60)
    print(f"OVERALL: {correct} correct, {partial} partial, {incorrect} incorrect out of {total}")
    print(f"SCORE: {score:.1f}%")
    print("=" * 60)

    # Breakdown by document
    doc_ranges = {
        "AlexNet": (1, 25),
        "Attention": (26, 55),
        "IPCC": (56, 80),
        "NASA Artemis": (81, 100),
    }

    print("\nBREAKDOWN BY DOCUMENT:")
    for doc_name, (start, end) in doc_ranges.items():
        doc_results = [r for r in results if start <= r["id"] <= end]
        if not doc_results:
            continue
        doc_correct = sum(1 for r in doc_results if r["grade"] == "CORRECT")
        doc_partial = sum(1 for r in doc_results if r["grade"] == "PARTIAL")
        doc_total = len(doc_results)
        doc_score = (doc_correct + 0.5 * doc_partial) / doc_total * 100
        print(f"  {doc_name:15s}: {doc_score:5.1f}% ({doc_correct}C / {doc_partial}P / {doc_total - doc_correct - doc_partial}I)")

    # Follow-up pair results
    follow_up_results = [r for r in results if r["is_follow_up"]]
    if follow_up_results:
        fu_correct = sum(1 for r in follow_up_results if r["grade"] == "CORRECT")
        print(f"\n  Follow-up accuracy: {fu_correct}/{len(follow_up_results)}")

    # List incorrect questions
    incorrect_list = [r for r in results if r["grade"] == "INCORRECT"]
    if incorrect_list:
        print(f"\nINCORRECT QUESTIONS ({len(incorrect_list)}):")
        for r in incorrect_list:
            print(f"  Q{r['id']}: {r['question']}")
            print(f"    Expected: {r['expected'][:80]}...")
            print(f"    Got:      {r['rag_answer'][:80]}...")
            print()


# --- MAIN ---

if __name__ == "__main__":
    print("Building RAG chain...")
    rag_chain = build_chain()

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    grader = create_grader()

    print("\n--- Evaluating standalone questions ---")
    standalone_results = evaluate_standalone(conversational_chain, grader)

    print("\n--- Evaluating follow-up pairs ---")
    followup_results = evaluate_follow_ups(conversational_chain, grader)

    all_results = standalone_results + followup_results
    print_report(all_results)

    # Save raw results
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\nRaw results saved to eval_results.json")
