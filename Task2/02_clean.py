from pathlib import Path

src = Path("raw_pages")
dst = Path("clean_pages")
dst.mkdir(exist_ok=True)

for file in src.glob("*.txt"):
    lines = []
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if len(line) < 40:
            continue
        if "http" in line:
            continue
        lines.append(line)

    dst_file = dst / file.name
    dst_file.write_text("\n".join(lines), encoding="utf-8")
