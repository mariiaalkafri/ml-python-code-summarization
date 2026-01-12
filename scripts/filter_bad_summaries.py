import json, os, re

BAD_ENDINGS = set(["of", "that", "is", "to", "and", "or", "with", "for", "in", "on", "as", "by"])
WS_RE = re.compile(r"\s+")

def ok_summary(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    s = WS_RE.sub(" ", s)
    words = s.split()
    if len(words) < 3:
        return False
    # ends with incomplete connector word
    if words[-1].lower().strip(".,;:") in BAD_ENDINGS:
        return False
    # too much punctuation-only / junk
    if sum(ch.isalnum() for ch in s) < 5:
        return False
    return True

def filter_file(inp, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    kept = 0
    dropped = 0
    with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            if ok_summary(ex.get("summary", "")):
                g.write(json.dumps(ex, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1
    print(f"{inp} -> {out} | kept={kept}, dropped={dropped}")

def main():
    # filter the MIX files
    filter_file("data/processed/train_mix_v2.jsonl", "data/processed/train_mix_v2_filtered.jsonl")
    filter_file("data/processed/valid_mix_v2.jsonl", "data/processed/valid_mix_v2_filtered.jsonl")

    # keep clean-only validation too (for “hard” evaluation)
    filter_file("data/processed/valid_clean_v2.jsonl", "data/processed/valid_clean_v2_filtered.jsonl")

if __name__ == "__main__":
    main()
