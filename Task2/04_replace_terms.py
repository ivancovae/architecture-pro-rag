import json
from pathlib import Path
import re

with open("terms_map.json", encoding="utf-8") as f:
    terms = json.load(f)

src = Path("clean_pages")
dst = Path("knowledge_base")
dst.mkdir(exist_ok=True)

sorted_terms = sorted(terms.items(), key=lambda x: len(x[0]), reverse=True)

for file in src.glob("*.txt"):
    text = file.read_text(encoding="utf-8")

    for original, fake in sorted_terms:
        text = re.sub(rf"\b{re.escape(original)}\b", fake, text, flags=re.IGNORECASE)

    dst_file = dst / file.name
    dst_file.write_text(text, encoding="utf-8")
