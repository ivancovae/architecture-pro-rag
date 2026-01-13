import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Загрузка индекса и метаданных
index = faiss.read_index("faiss.index")
term_name = "Caesennius Dibble"

with open("chunks_meta.json", encoding="utf-8") as f:
    metas = json.load(f)

with open("chunks.jsonl", encoding="utf-8") as f:
    texts = [json.loads(line)["text"] for line in f]

# Модель эмбеддингов
model = SentenceTransformer("all-MiniLM-L6-v2")

query = "What is " + term_name + " and how is it used"
query_emb = model.encode([query], convert_to_numpy=True)

# Поиск
D, I = index.search(query_emb, k=3)

print("Результаты поиска:\n")

for rank, idx in enumerate(I[0], start=1):
    print(f"#{rank}")
    print("Источник:", metas[idx]["source"])
    print("Chunk ID:", metas[idx]["chunk_id"])
    print(texts[idx][:300], "...\n")
