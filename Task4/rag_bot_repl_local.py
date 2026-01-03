import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama


TASK3_DIR = Path("../task3")
FAISS_INDEX_PATH = TASK3_DIR / "faiss.index"
CHUNKS_JSONL_PATH = TASK3_DIR / "chunks.jsonl"
META_JSON_PATH = TASK3_DIR / "chunks_meta.json"

MODEL_PATH = Path("models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")


TOP_K = 5
UNKNOWN_THRESHOLD = 1.2


FEWSHOT_QUERIES = [
    "What is Caesennius Dibble?",
    "Who are Chizu Gehreke?"
]
FEWSHOT_K = 2


def load_chunks_and_meta():
    with open(CHUNKS_JSONL_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line)["text"] for line in f]
    with open(META_JSON_PATH, encoding="utf-8") as f:
        metas = json.load(f)
    return chunks, metas


def retrieve(index, emb_model, query: str, chunks, metas, top_k: int = 5):
    q_emb = emb_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k=top_k)

    results = []
    for dist, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0:
            continue
        results.append({
            "distance": float(dist),
            "text": chunks[idx],
            "source": metas[idx].get("source"),
            "chunk_id": metas[idx].get("chunk_id"),
            "global_id": int(idx),
        })
    return results


def build_fewshot_block(index, emb_model, chunks, metas):
    examples = []
    for q in FEWSHOT_QUERIES:
        hits = retrieve(index, emb_model, q, chunks, metas, top_k=1)
        if not hits:
            continue
        h = hits[0]
        # few-shot ответ опирается на retrieved чанк (без фантазии)
        examples.append(
            f"Q: {q}\n"
            f"A: Based on the knowledge base: {h['text'][:220].strip()} ... "
            f"(source: {h['source']}#{h['chunk_id']})"
        )
        if len(examples) >= FEWSHOT_K:
            break
    return "\n\n".join(examples)


def build_prompt(user_query: str, retrieved, fewshot_block: str):
    # CoT: просим объяснять шаги, но кратко (2–4 пункта)
    system = (
        "You are a RAG assistant.\n"
        "Rules:\n"
        "1) Answer ONLY using the provided CONTEXT.\n"
        "2) If the answer is not in the context, reply exactly: I don't know\n"
        "3) Always output in this format:\n"
        "   Steps:\n"
        "   - ... (2-4 short bullets)\n"
        "   Answer: ...\n"
        "   Sources: file#chunk_id, ...\n"
        "Do not use outside knowledge. Do not invent facts.\n"
    )

    context_blocks = []
    sources = []
    for r in retrieved:
        tag = f"{r['source']}#{r['chunk_id']}"
        sources.append(tag)
        context_blocks.append(f"[{tag}]\n{r['text']}")

    context = "\n\n".join(context_blocks)

    user = (
        f"FEW-SHOT EXAMPLES:\n{fewshot_block}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"USER QUESTION:\n{user_query}\n"
    )

    return system, user, sources


def llm_answer(llm: Llama, system_prompt: str, user_prompt: str):
    prompt = (
    "[SYSTEM]\n"
    f"{system_prompt}\n"
    "[USER]\n"
    f"{user_prompt}\n"
    "[ASSISTANT]\n"
    )

    out = llm(prompt, max_tokens=512, temperature=0.2, stop=["</s>"])
    return out["choices"][0]["text"].strip()


def main():
    # 1) Загрузка индекса и данных
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    chunks, metas = load_chunks_and_meta()

    # 2) Эмбеддинг-модель (та же, что в task3)
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3) LLM (локально)
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=4096)

    # 4) Few-shot блок (из retrieval)
    fewshot_block = build_fewshot_block(index, emb_model, chunks, metas)

    print("Local RAG REPL bot is ready. Type a question, or 'exit'.\n")

    while True:
        q = input("You> ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        # 5) Retrieval
        retrieved = retrieve(index, emb_model, q, chunks, metas, top_k=TOP_K)

        # 6) Guardrail: если нет релевантного контекста — честно “I don't know”
        if not retrieved or retrieved[0]["distance"] > UNKNOWN_THRESHOLD:
            print("\nBot> I don't know\n")
            continue

        # 7) Prompt assembly (few-shot + CoT steps + sources)
        system, user, _ = build_prompt(q, retrieved, fewshot_block)

        # 8) Generation
        ans = llm_answer(llm, system, user)
        print(f"\nBot> {ans}\n")


if __name__ == "__main__":
    main()
