# scripts/make_clean_splits.py
import json, os, re

# Remove docstring immediately after a def header (best-effort, robust)
LEADING_DOCSTRING_RE = re.compile(
    r'(?s)(^\s*def\s+\w+\s*\(.*?\)\s*:\s*)([ \t]*("""|\'\'\').*?\3\s*)',
    re.M
)

# Remove full-line and inline # comments (conservative)
INLINE_HASH_RE = re.compile(r"(?m)([^\"']*)#.*$")  # avoids obvious string cases a bit

def strip_leading_docstring(code: str) -> str:
    return LEADING_DOCSTRING_RE.sub(r"\1", code, count=1)

def strip_hash_comments(code: str) -> str:
    out_lines = []
    for line in code.splitlines():
        # keep indentation and code; remove trailing # comment roughly
        m = INLINE_HASH_RE.match(line)
        out_lines.append(m.group(1).rstrip() if m else line)
    return "\n".join(out_lines)

def cleanup_blank_lines(code: str) -> str:
    lines = [ln.rstrip() for ln in code.splitlines()]
    # remove empty lines
    lines = [ln for ln in lines if ln.strip() != ""]
    return "\n".join(lines).strip()

def clean_code(code: str) -> str:
    code = strip_leading_docstring(code)
    code = strip_hash_comments(code)
    code = cleanup_blank_lines(code)
    return code

def convert_file(inp, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    kept = 0
    dropped = 0
    with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
        for line in f:
            ex = json.loads(line)
            code = ex.get("code", "")
            summ = ex.get("summary", "")
            if not code or not summ:
                dropped += 1
                continue
            ex["code"] = clean_code(code)
            if not ex["code"].strip():
                dropped += 1
                continue
            g.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1
    print(f"{inp} -> {out} | kept={kept} dropped={dropped}")

def main():
    convert_file("data/processed/train.jsonl", "data/processed/train_clean.jsonl")
    convert_file("data/processed/valid.jsonl", "data/processed/valid_clean.jsonl")
    convert_file("data/processed/test.jsonl",  "data/processed/test_clean.jsonl")

if __name__ == "__main__":
    main()
