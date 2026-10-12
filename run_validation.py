import sys
sys.stdout.reconfigure(encoding='utf-8')
import rag

test_questions = [
    "What subjects are present in my timetable?",
    "What do I have on Monday?",
    "What topics are covered in the syllabus?",
    "What are the important dates in the academic calendar?",
    "Tell me something that is not present in the documents."
]

print("==================================================")
print("ACADEMIC DOCUMENT ASSISTANT - VALIDATION TEST SUITE")
print("==================================================\n")

for i, q in enumerate(test_questions, 1):
    print(f"[{i}] Question: {q}")
    res = rag.ask_assistant(q, top_k=4)
    print(f"Answer:\n{res['answer'].strip()}")
    print(f"Sources: {res['sources']}")
    print("-" * 50 + "\n")
