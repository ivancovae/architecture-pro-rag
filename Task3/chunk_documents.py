from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

KB_PATH = Path("../task2/knowledge_base")
OUT_FILE = "chunks.jsonl"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

chunk_id_global = 0

with open(OUT_FILE, "w", encoding="utf-8") as out:
    for file in KB_PATH.glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)

        for local_id, chunk in enumerate(chunks):
            record = {
                "id": chunk_id_global,
                "text": chunk,
                "metadata": {
                    "source": file.name,
                    "chunk_id": local_id
                }
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            chunk_id_global += 1

print(f"Чанкинг завершён. Создано чанков: {chunk_id_global}")
print(f"Файл сохранён: {OUT_FILE}")
