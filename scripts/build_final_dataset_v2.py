import json, os, random, re
from typing import List, Dict

# ---------- CLEANING ----------
TRIPLE_BLOCK_RE = re.compile(r"(?s)('''.*?'''|\"\"\".*?\"\"\")")
FULL_LINE_COMMENT_RE = re.compile(r"(?m)^\s*#.*$")
INLINE_HASH_RE = re.compile(r"(?m)(?P<code>[^\"'\n]*)(#.*)$")
SINGLE_QUOTED_RE = re.compile(r"(?s)'([^'\\]|\\.)*'")
DOUBLE_QUOTED_RE = re.compile(r'(?s)"([^"\\]|\\.)*"')

def strip_docstrings_and_comments(code: str) -> str:
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

    # remove blank lines
    lines = [ln.rstrip() for ln in code.splitlines() if ln.strip() != ""]
    return "\n".join(lines).strip()

def strong_clean(code: str) -> str:
    # remove docstrings/comments + replace all strings with <STR>
    code = strip_docstrings_and_comments(code)
    code = SINGLE_QUOTED_RE.sub("<STR>", code)
    code = DOUBLE_QUOTED_RE.sub("<STR>", code)
    return code.strip()

def light_clean(code: str) -> str:
    # remove docstrings/comments only (keep normal strings)
    return strip_docstrings_and_comments(code)

# ---------- SUMMARY NORMALIZATION ----------
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
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    s = WS_RE.sub(" ", lines[0]).strip()
    m = FIRST_SENT_RE.match(s)
    if m:
        s = m.group(1).strip()
    return s

def is_good_summary(s: str) -> bool:
    if not s:
        return False
    s = s.strip()

    # kill obvious truncation signals
    if s.endswith("-"):
        return False
    if s.endswith("..."):
        return False

    words = s.split()
    if len(words) < 3:
        return False

    last = words[-1].lower().strip(".,;:")
    if last in BAD_ENDINGS:
        return False

    # if long and no punctuation end => likely truncated
    if len(words) >= 8 and s[-1] not in ".!?":
        return False

    # too little alnum
    if sum(ch.isalnum() for ch in s) < 8:
        return False

    # "protein-pro" style truncation after hyphen
    if "-" in s and len(words[-1]) <= 3 and s[-1].isalnum():
        # example: protein-pro
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

# ---------- BUILD ----------
def build_split(rows: List[Dict], mode: str) -> List[Dict]:
    """
    mode:
      - "strong": remove docstrings/comments + replace strings -> <STR>
      - "light":  remove docstrings/comments only
    """
    out = []
    for ex in rows:
        code = ex.get("code","")
        summ = ex.get("summary","")
        if not code or not summ:
            continue

        summ2 = normalize_summary(summ)
        if not is_good_summary(summ2):
            continue

        if mode == "strong":
            code2 = strong_clean(code)
        elif mode == "light":
            code2 = light_clean(code)
        else:
            raise ValueError("mode must be strong or light")

        if not code2:
            continue

        out.append({"code": code2, "summary": summ2})
    return out

def weighted_mix(clean70: List[Dict], light30: List[Dict], clean_ratio: float, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    # keep size = size of light30 base (original split size proxy)
    total = len(light30)
    n_clean = int(total * clean_ratio)
    n_light = total - n_clean

    random.shuffle(clean70)
    random.shuffle(light30)

    take_clean = clean70[:min(n_clean, len(clean70))]
    remaining = total - len(take_clean)
    take_light = light30[:min(remaining, len(light30))]

    mixed = take_clean + take_light
    random.shuffle(mixed)
    return mixed

def main():
    CLEAN_RATIO = 0.70
    SEED = 42

    train = read_jsonl("data/processed/train.jsonl")
    valid = read_jsonl("data/processed/valid.jsonl")
    test  = read_jsonl("data/processed/test.jsonl")

    # Build STRONG clean sets (for 70%)
    train_strong = build_split(train, mode="strong")
    valid_strong = build_split(valid, mode="strong")
    test_strong  = build_split(test,  mode="strong")

    # Build LIGHT clean sets (for 30%) — no docstrings/comments, but keep strings
    train_light = build_split(train, mode="light")
    valid_light = build_split(valid, mode="light")

    # Save clean-only (hard eval)
    write_jsonl("data/processed/train_clean_strong_final.jsonl", train_strong)
    write_jsonl("data/processed/valid_clean_strong_final.jsonl", valid_strong)
    write_jsonl("data/processed/test_clean_strong_final.jsonl",  test_strong)

    # Save mixed final (70% strong + 30% light)
    train_mix = weighted_mix(train_strong, train_light, clean_ratio=CLEAN_RATIO, seed=SEED)
    valid_mix = weighted_mix(valid_strong, valid_light, clean_ratio=CLEAN_RATIO, seed=SEED)

    write_jsonl("data/processed/train_mix_70strong_30light_final.jsonl", train_mix)
    write_jsonl("data/processed/valid_mix_70strong_30light_final.jsonl", valid_mix)

    print("DONE ✅")
    print("Saved:")
    print("  data/processed/train_clean_strong_final.jsonl")
    print("  data/processed/valid_clean_strong_final.jsonl")
    print("  data/processed/test_clean_strong_final.jsonl")
    print("  data/processed/train_mix_70strong_30light_final.jsonl")
    print("  data/processed/valid_mix_70strong_30light_final.jsonl")
    print(f"Counts: train_strong={len(train_strong)}, train_light={len(train_light)}, train_mix={len(train_mix)}")
    print(f"Counts: valid_strong={len(valid_strong)}, valid_light={len(valid_light)}, valid_mix={len(valid_mix)}")

if __name__ == "__main__":
    main()
