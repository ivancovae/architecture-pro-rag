from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()
BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("TG_BOT_TOKEN is not set")

TASK3_DIR = Path("../Task3")
print("\nTASK3_DIR=" + str(TASK3_DIR))
FAISS_INDEX = TASK3_DIR / "faiss.index"
print("\nFAISS_INDEX=" + str(FAISS_INDEX))
CHUNKS_JSONL = TASK3_DIR / "chunks.jsonl"
print("\nCHUNKS_JSONL=" + str(CHUNKS_JSONL))
META_JSON = TASK3_DIR / "chunks_meta.json"
print("\nMETA_JSON=" + str(META_JSON))

MODEL_PATH = os.path.abspath(str(Path("../Task4/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")))
print("\nMODEL_PATH=" + str(MODEL_PATH))

TOP_K = 5
UNKNOWN_THRESHOLD = 1.2
