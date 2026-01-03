from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()
BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("TG_BOT_TOKEN is not set")

TASK3_DIR = Path("../task3")
FAISS_INDEX = TASK3_DIR / "faiss.index"
CHUNKS_JSONL = TASK3_DIR / "chunks.jsonl"
META_JSON = TASK3_DIR / "chunks_meta.json"

MODEL_PATH = Path("models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")

TOP_K = 5
UNKNOWN_THRESHOLD = 1.2
