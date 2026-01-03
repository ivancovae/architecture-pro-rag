import faiss
import numpy as np
import json
from pathlib import Path

EMB_FILE = Path("embeddings.npy")
META_FILE = Path("chunks_meta.json")
INDEX_FILE = Path("faiss.index")

# Загружаем эмбеддинги
embeddings = np.load(EMB_FILE)

dim = embeddings.shape[1]
print(f"Загружено эмбеддингов: {embeddings.shape[0]}")
print(f"Размерность: {dim}")

# Создаём FAISS индекс (точный поиск, L2)
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print(f"В индексе векторов: {index.ntotal}")

# Сохраняем индекс
faiss.write_index(index, str(INDEX_FILE))

print(f"FAISS индекс сохранён: {INDEX_FILE}")
