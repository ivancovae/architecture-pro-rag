import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

CHUNKS_FILE = Path("chunks.jsonl")
EMB_FILE = Path("embeddings.npy")
META_FILE = Path("chunks_meta.json")

# Загружаем модель эмбеддингов
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
metas = []

# Читаем чанки
with open(CHUNKS_FILE, encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        texts.append(record["text"])
        metas.append(record["metadata"])

print(f"Загружено чанков: {len(texts)}")

# Генерация эмбеддингов
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=False
)

# Сохраняем результаты
np.save(EMB_FILE, embeddings)

with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(metas, f, ensure_ascii=False, indent=2)

print("Эмбеддинги успешно сгенерированы")
print(f"Форма embeddings: {embeddings.shape}")
print(f"Файлы сохранены: {EMB_FILE}, {META_FILE}")
