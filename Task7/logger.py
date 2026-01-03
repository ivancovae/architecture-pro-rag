import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class RAGQueryLogger:
    """
    Логирование запросов и ответов локального RAG-бота (FAISS-based).
    """

    def __init__(self, log_path: Optional[Path] = None):
        base_dir = Path(__file__).resolve().parent
        self.log_path = log_path or base_dir / "logs.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict],
        success: Optional[bool] = None,
    ) -> Dict:
        """
        Сохраняет один запрос в лог.
        """

        if success is None:
            success = self._detect_success(answer, retrieved_chunks)

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "answer_length": len(answer),
            "chunks_found": bool(retrieved_chunks),
            "chunks_count": len(retrieved_chunks),
            "sources": [
                {
                    "source": c.get("source"),
                    "chunk_id": c.get("chunk_id"),
                    "distance": c.get("distance"),
                }
                for c in retrieved_chunks
            ],
            "success": success,
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record

    def _detect_success(self, answer: str, chunks: List[Dict]) -> bool:
        """
        Простая эвристика успешного ответа.
        """
        failure_markers = [
            "i don't know",
            "не знаю",
            "нет информации",
            "not found",
        ]

        text = answer.lower()

        if any(marker in text for marker in failure_markers):
            return False
        if not chunks:
            return False
        if len(answer) < 30:
            return False

        return True

    def read_all(self) -> List[Dict]:
        if not self.log_path.exists():
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def clear(self):
        if self.log_path.exists():
            self.log_path.unlink()
