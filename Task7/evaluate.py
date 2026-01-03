import sys
import time
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from task4.rag_core import answer_query
from logger import RAGQueryLogger


QUESTIONS_FILE = BASE_DIR / "golden_questions.txt"
SUMMARY_FILE = BASE_DIR / "evaluation_summary.txt"


def load_questions(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    questions = load_questions(QUESTIONS_FILE)
    logger = RAGQueryLogger()

    stats = {
        "total": 0,
        "success": 0,
        "fail": 0,
        "no_chunks": 0,
    }

    start = time.time()

    for q in questions:
        result = answer_query(q)

        answer = result.get("answer", "")
        chunks = result.get("chunks", [])

        record = logger.write(
            query=q,
            answer=answer,
            retrieved_chunks=chunks,
        )

        stats["total"] += 1
        if record["success"]:
            stats["success"] += 1
        else:
            stats["fail"] += 1

        if not record["chunks_found"]:
            stats["no_chunks"] += 1

        print(f"[✓] {q}")

    elapsed = round(time.time() - start, 2)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(f"Evaluation time: {datetime.now().isoformat()}\n")
        f.write(f"Total questions: {stats['total']}\n")
        f.write(f"Successful answers: {stats['success']}\n")
        f.write(f"Failed answers: {stats['fail']}\n")
        f.write(f"No chunks found: {stats['no_chunks']}\n")
        f.write(f"Execution time: {elapsed}s\n")

    print("\nEvaluation finished")
    print(stats)


if __name__ == "__main__":
    main()
