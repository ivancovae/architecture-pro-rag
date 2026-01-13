import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# OpenAI Responses API (рекомендуемый)
from openai import OpenAI

# --------- Paths (подстрой под свою структуру) ----------
# если task4 рядом с task3:
TASK3_DIR = os.path.join("..", "task3")
FAISS_INDEX_PATH = os.path.join(TASK3_DIR, "faiss.index")
CHUNKS_JSONL_PATH = os.path.join(TASK3_DIR, "chunks.jsonl")
META_JSON_PATH = os.path.join(TASK3_DIR, "chunks_meta.json")

# --------- Retrieval settings ----------
TOP_K = 5

# Порог "Я не знаю" (для IndexFlatL2 ниже = ближе).
# Точный порог зависит от данных; стартово поставим 1.2 и при необходимости подстроим.
UNKNOWN_THRESHOLD = 1.2

# --------- Few-shot settings ----------
FEWSHOT_K = 2  # 1-2 примера
FEWSHOT_QUERIES = [
    "What is Caesennius Dibble?",
    "Who are Chizu Gehreke?"
]

def load_chunks_and_meta(chunks_jsonl_path: str, meta_json_path: str):
    with open(chunks_jsonl_path, encoding="utf-8") as f:
        chunks = [json.loads(line)["text"] for line in f]
    with open(meta_json_path, encoding="utf-8") as f:
        metas = json.load(f)
    return chunks, metas

def retrieve(index, model, query: str, chunks, metas, top_k: int = 5):
    q_emb = model.encode([query], convert_to_numpy=True)
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
            "global_id": idx
        })
    return results

def build_fewshot_block(index, emb_model, chunks, metas):
    # Few-shot должен быть "из той же предметной области".
    # Мы делаем это правильно: вытягиваем контекст из индекса и формируем примеры на основе retrieved-чанков.
    examples = []
    for q in FEWSHOT_QUERIES:
        hits = retrieve(index, emb_model, q, chunks, metas, top_k=1)
        if not hits:
            continue
        h = hits[0]
        # Ответ в few-shot — краткое резюме факта из найденного чанка (а не фантазия).
        # Формулируем так, чтобы LLM копировала стиль: "опирайся на контекст + укажи источник".
        examples.append(
            f"Q: {q}\n"
            f"A: Based on the knowledge base, {h['text'][:220].strip()} ... "
            f"(source: {h['source']}#{h['chunk_id']})"
        )
        if len(examples) >= FEWSHOT_K:
            break
    return "\n\n".join(examples)

def make_prompt(user_query: str, retrieved, fewshot_block: str):
    # CoT: мы не просим раскрывать «внутренние рассуждения», а просим краткое объяснение шагов.
    # Это даст проверяемый "CoT-like" вывод без лишней детализации.
    system = (
        "You are a RAG assistant. You must answer ONLY using the provided CONTEXT.\n"
        "If the context does not contain the answer, say exactly: \"I don't know\".\n"
        "Output format:\n"
        "1) Reasoning (brief): 2-4 short bullet points describing what you looked up and why.\n"
        "2) Answer: the final answer.\n"
        "3) Sources: list of sources in form filename#chunk_id.\n"
        "Do not invent facts. Do not use outside knowledge.\n"
    )

    context_lines = []
    sources = []
    for r in retrieved:
        context_lines.append(
            f"[{r['source']}#{r['chunk_id']}] {r['text']}"
        )
        sources.append(f"{r['source']}#{r['chunk_id']}")

    context = "\n\n".join(context_lines)

    user = (
        f"FEW-SHOT EXAMPLES (from the same knowledge base):\n"
        f"{fewshot_block}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"USER QUESTION:\n{user_query}\n"
    )
    return system, user, sources

def main():
    # Load index + data
    index = faiss.read_index(FAISS_INDEX_PATH)
    chunks, metas = load_chunks_and_meta(CHUNKS_JSONL_PATH, META_JSON_PATH)

    emb_model = SentenceTransformer("all-MiniLM-L6-v2")

    # OpenAI client
    # Требуется переменная окружения OPENAI_API_KEY
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    fewshot_block = build_fewshot_block(index, emb_model, chunks, metas)

    print("RAG REPL bot is ready. Type a question, or 'exit'.\n")

    while True:
        q = input("You> ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        retrieved = retrieve(index, emb_model, q, chunks, metas, top_k=TOP_K)

        # Если лучший результат слишком далеко — честно говорим "I don't know"
        if not retrieved or retrieved[0]["distance"] > UNKNOWN_THRESHOLD:
            print("\nBot> I don't know\n")
            continue

        system, user, _ = make_prompt(q, retrieved, fewshot_block)

        # Responses API: рекомендован для новых проектов :contentReference[oaicite:1]{index=1}
        resp = client.responses.create(
            model="gpt-5-mini",
            instructions=system,
            input=user,
        )

        # В Responses API текст можно собрать так:
        # В SDK обычно есть resp.output_text, но оставим более совместимый путь:
        out_text = getattr(resp, "output_text", None)
        if out_text is None:
            # fallback: попробуем извлечь текст из output
            out_text = ""
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", "") == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", "") == "output_text":
                            out_text += getattr(c, "text", "")

        print(f"\nBot> {out_text}\n")

if __name__ == "__main__":
    main()
