import json, os, re

WS_RE = re.compile(r"\s+")
FIRST_SENT_RE = re.compile(r"^(.+?[.!?])(\s|$)")

# kill weird docstring prefixes
DOCSTRING_PREFIX_RE = re.compile(r'^(r|u|ur|ru)?("""|\'\'\')\s*', re.IGNORECASE)
LEADING_QUOTES_RE = re.compile(r'^[\'"]+')
import json, os, re

WS_RE = re.compile(r"\s+")
FIRST_SENT_RE = re.compile(r"^(.+?[.!?])(\s|$)")

# kill weird docstring prefixes
DOCSTRING_PREFIX_RE = re.compile(r'^(r|u|ur|ru)?("""|\'\'\')\s*', re.IGNORECASE)
LEADING_QUOTES_RE = re.compile(r'^[\'"]+')

# CLI/help patterns to drop
CLI_PATTERN_RE = re.compile(r"^%prog\b|^usage:\b|^-h\b|^--help\b", re.IGNORECASE)

BAD_ENDINGS = {
    "a","an","the","of","that","this","these","those","is","are","was","were",
    "to","and","or","with","for","in","on","as","by","from","at","into","over",
    "if","when","while","which","who","whom","whose","it","its","be","been","being"
}

def normalize_summary(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""

    # first non-empty line
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    s = lines[0]

    s = WS_RE.sub(" ", s).strip()

    # remove docstring markers like r""" or """
    s = DOCSTRING_PREFIX_RE.sub("", s).strip()
    s = LEADING_QUOTES_RE.sub("", s).strip()

    # keep first sentence if punctuation exists
    m = FIRST_SENT_RE.match(s)
    if m:
        s = m.group(1).strip()

    # if it’s a short phrase without punctuation, keep as-is but strip trailing junk
    return s.strip()

def is_good_summary(s: str) -> bool:
    if not s:
        return False

    # drop CLI/help strings
    if CLI_PATTERN_RE.search(s):
        return False

    words = s.split()
    if len(words) < 3:
        return False

    last = words[-1].lower().strip(".,;:")
    if last in BAD_ENDINGS:
        return False

    # likely truncated if long and no punctuation
    if len(words) >= 8 and s[-1] not in ".!?":
        return False

    # remove weird remaining triple quotes
    if '"""' in s or "'''" in s:
        return False

    # must contain some alnum
    if sum(ch.isalnum() for ch in s) < 8:
        return False

    return True

def sanitize_file(inp: str, out: str):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    kept = 0
    dropped = 0

    with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            code = ex.get("code","")
            summ = ex.get("summary","")
            if not code or not summ:
                dropped += 1
                continue

            summ2 = normalize_summary(summ)

            # optional: add a period for short phrases (nice for consistency)
            if summ2 and summ2[-1] not in ".!?" and len(summ2.split()) <= 6:
                summ2 = summ2 + "."

            if not is_good_summary(summ2):
                dropped += 1
                continue

            ex["summary"] = summ2
            g.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1

    print(f"{inp} -> {out} | kept={kept}, dropped={dropped}")

def main():
    # sanitize the FINAL datasets you will train on
    sanitize_file(
        "data/processed/train_mix_70strong_30light_final.jsonl",
        "data/processed/train_mix_70strong_30light_final_sanitized.jsonl"
    )
    sanitize_file(
        "data/processed/valid_mix_70strong_30light_final.jsonl",
        "data/processed/valid_mix_70strong_30light_final_sanitized.jsonl"
    )
    sanitize_file(
        "data/processed/valid_clean_strong_final.jsonl",
        "data/processed/valid_clean_strong_final_sanitized.jsonl"
    )

if __name__ == "__main__":
    main()

# CLI/help patterns to drop
CLI_PATTERN_RE = re.compile(r"^%prog\b|^usage:\b|^-h\b|^--help\b", re.IGNORECASE)

BAD_ENDINGS = {
    "a","an","the","of","that","this","these","those","is","are","was","were",
    "to","and","or","with","for","in","on","as","by","from","at","into","over",
    "if","when","while","which","who","whom","whose","it","its","be","been","being"
}

def normalize_summary(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""

    # first non-empty line
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""
    s = lines[0]

    s = WS_RE.sub(" ", s).strip()

    # remove docstring markers like r""" or """
    s = DOCSTRING_PREFIX_RE.sub("", s).strip()
    s = LEADING_QUOTES_RE.sub("", s).strip()

    # keep first sentence if punctuation exists
    m = FIRST_SENT_RE.match(s)
    if m:
        s = m.group(1).strip()

    # if it’s a short phrase without punctuation, keep as-is but strip trailing junk
    return s.strip()

def is_good_summary(s: str) -> bool:
    if not s:
        return False

    # drop CLI/help strings
    if CLI_PATTERN_RE.search(s):
        return False

    words = s.split()
    if len(words) < 3:
        return False

    last = words[-1].lower().strip(".,;:")
    if last in BAD_ENDINGS:
        return False

    # likely truncated if long and no punctuation
    if len(words) >= 8 and s[-1] not in ".!?":
        return False

    # remove weird remaining triple quotes
    if '"""' in s or "'''" in s:
        return False

    # must contain some alnum
    if sum(ch.isalnum() for ch in s) < 8:
        return False

    return True

def sanitize_file(inp: str, out: str):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    kept = 0
    dropped = 0

    with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            code = ex.get("code","")
            summ = ex.get("summary","")
            if not code or not summ:
                dropped += 1
                continue

            summ2 = normalize_summary(summ)

            # optional: add a period for short phrases (nice for consistency)
            if summ2 and summ2[-1] not in ".!?" and len(summ2.split()) <= 6:
                summ2 = summ2 + "."

            if not is_good_summary(summ2):
                dropped += 1
                continue

            ex["summary"] = summ2
            g.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1

    print(f"{inp} -> {out} | kept={kept}, dropped={dropped}")

def main():
    # sanitize the FINAL datasets you will train on
    sanitize_file(
        "data/processed/train_mix_70strong_30light_final.jsonl",
        "data/processed/train_mix_70strong_30light_final_sanitized.jsonl"
    )
    sanitize_file(
        "data/processed/valid_mix_70strong_30light_final.jsonl",
        "data/processed/valid_mix_70strong_30light_final_sanitized.jsonl"
    )
    sanitize_file(
        "data/processed/valid_clean_strong_final.jsonl",
        "data/processed/valid_clean_strong_final_sanitized.jsonl"
    )

if __name__ == "__main__":
    main()
