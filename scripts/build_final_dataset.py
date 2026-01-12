import json, os, random, re
from typing import List, Dict

# ---------- CLEANING CODE (remove docstrings/comments/strings) ----------
TRIPLE_BLOCK_RE = re.compile(r"(?s)('''.*?'''|\"\"\".*?\"\"\")")  # remove any triple-quoted blocks anywhere
FULL_LINE_COMMENT_RE = re.compile(r"(?m)^\s*#.*$")               # remove full comment lines
INLINE_HASH_RE = re.compile(r"(?m)(?P<code>[^\"'\n]*)(#.*)$")    # simple inline hash removal

# replace remaining string literals with <STR> (best-effort)
SINGLE_QUOTED_RE = re.compile(r"(?s)'([^'\\]|\\.)*'")
DOUBLE_QUOTED_RE = re.compile(r'(?s)"([^"\\]|\\.)*"')

def clean_code(code: str) -> str:
    code = (code or "")
    code = TRIPLE_BLOCK_RE.sub("", code)
    code = FULL_LINE_COMMENT_RE.sub("", code)

    out_lines = []
    for line in code.splitlines():
        m = INLINE_HASH_RE.match(line)
        if m and m.group("code").strip():
            out_lines.append(m.group("code").rstrip())
        else:
            out_lines.append(line.rstrip())
    code = "\n".join(out_lines)

    code = SINGLE_QUOTED_RE.sub("<STR>", code)
    code = DOUBLE_QUOTED_RE.sub("<STR>", code)

    # remove blank lines
    lines = [ln.rstrip() for ln in code.splitlines() if ln.strip() != ""]
    return "\n".join(lines).strip()

# ---------- SUMMARY NORMALIZATION (fix truncation) ----------
WS_RE = re.compile(r"\s+")
FIRST_SENT_RE = re.compile(r"^(.+?[.!?])(\s|$)")

BAD_ENDINGS = {
    "a","an","the","of","that","this","these","those","is","are","was","were",
    "to","and","or","with","for","in","on","as","by","from","at","into","over",
    "if","when","while","which","who","whom","whose","it","its","be","been","being"
}

def normalize_summary(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # take first non-empty line
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    s = lines[0]
    s = WS_RE.sub(" ", s).strip()

    # take first sentence if it ends with punctuation
    m = FIRST_SENT_RE.match(s)
    if m:
        s = m.group(1).strip()
    return s

def is_good_summary(s: str) -> bool:
    if not s:
        return False
    words = s.split()
    if len(words) < 3:
        return False

    last = words[-1].lower().strip(".,;:")
    if last in BAD_ENDINGS:
        return False

    # if long but no end punctuation, likely truncated
    if len(words) >= 8 and s[-1] not in ".!?":
        return False

    if sum(ch.isalnum() for ch in s) < 8:
        return False

    return True

# ---------- IO ----------
def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as g:
        for ex in rows:
            g.write(json.dumps(ex, ensure_ascii=False) + "\n")

# ---------- BUILD PIPELINE ----------
def build_clean(rows: List[Dict]) -> List[Dict]:
    out = []
    dropped = 0
    for ex in rows:
        code = ex.get("code","")
        summ = ex.get("summary","")
        if not code or not summ:
            dropped += 1
            continue
        summ2 = normalize_summary(summ)
        if not is_good_summary(summ2):
            dropped += 1
            continue
        code2 = clean_code(code)
        if not code2:
            dropped += 1
            continue
        out.append({"code": code2, "summary": summ2})
    return out

def build_orig_normalized(rows: List[Dict]) -> List[Dict]:
    out = []
    dropped = 0
    for ex in rows:
        code = ex.get("code","")
        summ = ex.get("summary","")
        if not code or not summ:
            dropped += 1
            continue
        summ2 = normalize_summary(summ)
        if not is_good_summary(summ2):
            dropped += 1
            continue
        out.append({"code": code, "summary": summ2})
    return out

def weighted_mix(clean_rows: List[Dict], orig_rows: List[Dict], clean_ratio: float, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    total = len(orig_rows)  # keep same size as original split
    n_clean = int(total * clean_ratio)
    n_orig = total - n_clean

    random.shuffle(clean_rows)
    random.shuffle(orig_rows)

    take_clean = clean_rows[:min(n_clean, len(clean_rows))]
    remaining = total - len(take_clean)
    take_orig = orig_rows[:min(remaining, len(orig_rows))]

    mixed = take_clean + take_orig
    random.shuffle(mixed)
    return mixed

def main():
    # SETTINGS
    CLEAN_RATIO = 0.70  # 70% clean, 30% original
    SEED = 42

    # INPUTS (from prepare_data.py)
    train_path = "data/processed/train.jsonl"
    valid_path = "data/processed/valid.jsonl"
    test_path  = "data/processed/test.jsonl"

    # OUTPUTS
    out_train_clean = "data/processed/train_clean_final.jsonl"
    out_valid_clean = "data/processed/valid_clean_final.jsonl"
    out_test_clean  = "data/processed/test_clean_final.jsonl"

    out_train_mix = "data/processed/train_mix_70c30o_final.jsonl"
    out_valid_mix = "data/processed/valid_mix_70c30o_final.jsonl"

    # LOAD
    train = read_jsonl(train_path)
    valid = read_jsonl(valid_path)
    test  = read_jsonl(test_path)

    # BUILD normalized original + cleaned
    train_orig_nf = build_orig_normalized(train)
    valid_orig_nf = build_orig_normalized(valid)

    train_clean = build_clean(train)
    valid_clean = build_clean(valid)
    test_clean  = build_clean(test)

    # SAVE clean-only (useful for “hard” evaluation)
    write_jsonl(out_train_clean, train_clean)
    write_jsonl(out_valid_clean, valid_clean)
    write_jsonl(out_test_clean, test_clean)

    # MIX 70/30 (size ~ original split, not doubled)
    train_mix = weighted_mix(train_clean, train_orig_nf, clean_ratio=CLEAN_RATIO, seed=SEED)
    valid_mix = weighted_mix(valid_clean, valid_orig_nf, clean_ratio=CLEAN_RATIO, seed=SEED)

    write_jsonl(out_train_mix, train_mix)
    write_jsonl(out_valid_mix, valid_mix)

    print("DONE ✅")
    print(f"Clean-only saved:")
    print(f"  {out_train_clean}  rows={len(train_clean)}")
    print(f"  {out_valid_clean}  rows={len(valid_clean)}")
    print(f"  {out_test_clean}   rows={len(test_clean)}")
    print(f"Mixed 70/30 saved (same size as original splits):")
    print(f"  {out_train_mix} rows={len(train_mix)} (70% clean / 30% orig)")
    print(f"  {out_valid_mix} rows={len(valid_mix)} (70% clean / 30% orig)")

if __name__ == "__main__":
    main()
