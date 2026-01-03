import json
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

from config import (
    FAISS_INDEX, CHUNKS_JSONL, META_JSON,
    MODEL_PATH, TOP_K, UNKNOWN_THRESHOLD
)

# ---- load once ----
index = faiss.read_index(str(FAISS_INDEX))
embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm = Llama(model_path=str(MODEL_PATH), n_ctx=4096)

with open(CHUNKS_JSONL, encoding="utf-8") as f:
    CHUNKS = [json.loads(l)["text"] for l in f]

with open(META_JSON, encoding="utf-8") as f:
    META = json.load(f)


def retrieve(query: str):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, TOP_K)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append({
            "dist": float(dist),
            "text": CHUNKS[idx],
            "src": f"{META[idx]['source']}#{META[idx]['chunk_id']}"
        })
    return results


def build_prompt(query, retrieved):
    context = "\n\n".join(
        f"[{r['src']}]\n{r['text']}" for r in retrieved
    )

    system = (
        "You are a RAG assistant.\n"
        "Rules:\n"
        "1) Use ONLY the CONTEXT.\n"
        "2) If answer not found — say exactly: I don't know.\n"
        "3) Explain reasoning briefly.\n"
        "Format:\n"
        "Steps:\n- ...\nAnswer: ...\nSources: ...\n"
    )

    user = f"""
CONTEXT:
{context}

QUESTION:
{query}
"""

    return system, user


def answer(query: str) -> str:
    retrieved = retrieve(query)

    if not retrieved or retrieved[0]["dist"] > UNKNOWN_THRESHOLD:
        return "I don't know"

    system, user = build_prompt(query, retrieved)

    prompt = f"[SYSTEM]\n{system}\n[USER]\n{user}\n[ASSISTANT]\n"

    out = llm(prompt, max_tokens=512, temperature=0.2, stop=["</s>"])
    return out["choices"][0]["text"].strip()
