import json
import time
import hashlib
from pathlib import Path
from datetime import datetime

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---- Paths ----
DOCS_DIR = Path("docs")
STATE_FILE = Path("state.json")
LOG_FILE = Path("update.log")

FAISS_INDEX = Path("../task3/faiss.index")

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500


# ---- Logging ----
def log(msg: str):
    line = f"[{datetime.now().isoformat()}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---- Utils ----
def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(path: Path, state: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def chunk_text(text: str, chunk_size: int):
    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
        if text[i:i + chunk_size].strip()
    ]


# ---- Main ----
def main():
    start = time.time()
    log("Index update started")

    # Load state and index
    state = load_state(STATE_FILE)
    emb_model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(str(FAISS_INDEX))

    files_scanned = 0
    files_updated = 0
    total_new_chunks = 0

    for file in DOCS_DIR.glob("*.*"):
        files_scanned += 1

        try:
            current_hash = file_hash(file)
        except Exception as e:
            log(f"Failed to read file {file.name}: {e}")
            continue

        if state.get(file.name) == current_hash:
            continue  # file unchanged

        text = file.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            log(f"Skipped empty file: {file.name}")
            state[file.name] = current_hash
            continue

        chunks = chunk_text(text, CHUNK_SIZE)
        if not chunks:
            log(f"No valid chunks extracted from: {file.name}")
            state[file.name] = current_hash
            continue

        embeddings = emb_model.encode(chunks, convert_to_numpy=True)

        # Safety: ensure correct shape
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        index.add(embeddings)

        files_updated += 1
        total_new_chunks += len(chunks)
        state[file.name] = current_hash

        log(f"Processed {file.name}: {len(chunks)} chunks added")

    # Save updated index and state
    faiss.write_index(index, str(FAISS_INDEX))
    save_state(STATE_FILE, state)

    elapsed = round(time.time() - start, 2)
    log(
        f"Index update finished | "
        f"scanned={files_scanned}, "
        f"updated={files_updated}, "
        f"new_chunks={total_new_chunks}, "
        f"index_size={index.ntotal}, "
        f"time={elapsed}s"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
