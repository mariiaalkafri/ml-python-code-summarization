import json, os, re

# 1) Remove ANY triple-quoted blocks anywhere (docstrings, long text, etc.)
TRIPLE_BLOCK_RE = re.compile(r"(?s)('''.*?'''|\"\"\".*?\"\"\")")

# 2) Remove full-line comments and inline comments (best-effort)
FULL_LINE_COMMENT_RE = re.compile(r"(?m)^\s*#.*$")
INLINE_COMMENT_RE = re.compile(r"(?m)(?P<code>[^\"'\n]*)(#.*)$")

# 3) Replace remaining normal string literals with <STR> (optional but recommended)
#    This is a best-effort regex; it won't be a perfect lexer but works well enough for ML data cleaning.
SINGLE_QUOTED_RE = re.compile(r"(?s)'([^'\\]|\\.)*'")
DOUBLE_QUOTED_RE = re.compile(r'(?s)"([^"\\]|\\.)*"')

def remove_triple_quotes(code: str) -> str:
    return TRIPLE_BLOCK_RE.sub("", code)

def remove_hash_comments(code: str) -> str:
    # remove full-line comments
    code = FULL_LINE_COMMENT_RE.sub("", code)

    # remove inline comments in a conservative way (won't catch all cases, but safe-ish)
    out = []
    for line in code.splitlines():
        # if line contains #, try to keep only the "code" part
        m = INLINE_COMMENT_RE.match(line)
        if m and m.group("code").strip():
            out.append(m.group("code").rstrip())
        else:
            # if it was only a comment or unmatched, keep line as-is
            out.append(line.rstrip())
    return "\n".join(out)

def replace_string_literals(code: str) -> str:
    # replace normal quoted strings with <STR>
    code = SINGLE_QUOTED_RE.sub("<STR>", code)
    code = DOUBLE_QUOTED_RE.sub("<STR>", code)
    return code

def cleanup(code: str) -> str:
    lines = [ln.rstrip() for ln in code.splitlines()]
    # remove empty lines
    lines = [ln for ln in lines if ln.strip() != ""]
    return "\n".join(lines).strip()

def clean_code(code: str, replace_strings: bool = True) -> str:
    code = remove_triple_quotes(code)
    code = remove_hash_comments(code)
    if replace_strings:
        code = replace_string_literals(code)
    code = cleanup(code)
    return code

def convert_file(inp, out, replace_strings=True):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    kept = 0
    dropped = 0

    with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            code = ex.get("code", "")
            summ = ex.get("summary", "")
            if not code or not summ:
                dropped += 1
                continue

            new_code = clean_code(code, replace_strings=replace_strings)
            if not new_code:
                dropped += 1
                continue

            ex["code"] = new_code
            g.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1

    print(f"{inp} -> {out} | kept={kept}, dropped={dropped}")

def main():
    convert_file("data/processed/train.jsonl", "data/processed/train_clean_v2.jsonl", replace_strings=True)
    convert_file("data/processed/valid.jsonl", "data/processed/valid_clean_v2.jsonl", replace_strings=True)
    convert_file("data/processed/test.jsonl",  "data/processed/test_clean_v2.jsonl",  replace_strings=True)

if __name__ == "__main__":
    main()
