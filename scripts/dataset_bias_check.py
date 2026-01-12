import json
import re
import argparse
from collections import Counter

TRIPLE_QUOTE_RE = re.compile(r"('{3}|\"{3})")
COMMENT_RE = re.compile(r"(?m)^\s*#")  # lines that start with #
DOCSTRING_BLOCK_RE = re.compile(r"(?s)^[ \t]*('{3}|\"{3})(.*?)(\1)")

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")  # simple identifier-ish tokens


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_first_docstring(code: str) -> str:
    """
    Best-effort: grabs a leading triple-quoted block if present (typical docstring).
    This is not a full Python parser, but it's good enough for dataset diagnostics.
    """
    m = DOCSTRING_BLOCK_RE.search(code)
    if not m:
        return ""
    return (m.group(2) or "").strip()


def tokenize(text: str):
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def jaccard(a_tokens, b_tokens):
    A, B = set(a_tokens), set(b_tokens)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="Path to FULL jsonl (250k)")
    ap.add_argument("--subset", required=True, help="Path to SUBSET jsonl (50k)")
    ap.add_argument("--copy_jaccard_threshold", type=float, default=0.65,
                    help="Docstring-summary Jaccard threshold to count as 'copy-like'")
    args = ap.parse_args()

    def compute_stats(path):
        c = Counter()
        jaccard_scores = []

        for ex in iter_jsonl(path):
            code = ex.get("code", "") or ""
            summary = ex.get("summary", "") or ""

            has_triple_quotes = bool(TRIPLE_QUOTE_RE.search(code))
            has_hash_comments = bool(COMMENT_RE.search(code))
            doc = extract_first_docstring(code)

            c["n"] += 1
            if has_triple_quotes:
                c["has_triple_quotes"] += 1
            if has_hash_comments:
                c["has_hash_comments"] += 1
            if doc:
                c["has_leading_docstring_block"] += 1

            # Copy-likeness: compare summary to extracted docstring (if any)
            if doc:
                s_tok = tokenize(summary)
                d_tok = tokenize(doc)
                js = jaccard(s_tok, d_tok)
                jaccard_scores.append(js)
                if js >= args.copy_jaccard_threshold:
                    c["copy_like_docstring_summary"] += 1
            else:
                c["no_docstring_for_copy_check"] += 1

        # summary of jaccard distribution (only where docstring exists)
        if jaccard_scores:
            jaccard_scores_sorted = sorted(jaccard_scores)
            def pct(p):
                idx = int(round((p/100)*(len(jaccard_scores_sorted)-1)))
                return jaccard_scores_sorted[idx]
            j_stats = {
                "count": len(jaccard_scores),
                "mean": sum(jaccard_scores)/len(jaccard_scores),
                "p50": pct(50),
                "p75": pct(75),
                "p90": pct(90),
            }
        else:
            j_stats = {"count": 0, "mean": None, "p50": None, "p75": None, "p90": None}

        return c, j_stats

    full_c, full_j = compute_stats(args.full)
    sub_c, sub_j = compute_stats(args.subset)

    def rate(x, n):
        return 0.0 if n == 0 else (100.0 * x / n)

    def print_block(name, c, j):
        n = c["n"]
        print(f"\n==== {name} ====")
        print(f"Samples: {n}")
        print(f'Has any triple quotes (""" or \'\'\'): {c["has_triple_quotes"]} ({rate(c["has_triple_quotes"], n):.2f}%)')
        print(f"Has # comment line: {c['has_hash_comments']} ({rate(c['has_hash_comments'], n):.2f}%)")
        print(f"Has leading docstring block: {c['has_leading_docstring_block']} ({rate(c['has_leading_docstring_block'], n):.2f}%)")
        denom = c["has_leading_docstring_block"]
        print(f"Copy-like (Jaccard >= {args.copy_jaccard_threshold}) among docstring-block cases: "
              f"{c['copy_like_docstring_summary']} ({rate(c['copy_like_docstring_summary'], denom):.2f}%)")
        print(f"Docstring-summary Jaccard stats (only where docstring block exists): {j}")

    print_block("FULL", full_c, full_j)
    print_block("SUBSET", sub_c, sub_j)

    # Quick “amplified bias?” hint:
    print("\n==== QUICK INTERPRETATION ====")
    full_doc_rate = rate(full_c["has_leading_docstring_block"], full_c["n"])
    sub_doc_rate = rate(sub_c["has_leading_docstring_block"], sub_c["n"])
    full_copy_rate = rate(full_c["copy_like_docstring_summary"], full_c["has_leading_docstring_block"])
    sub_copy_rate = rate(sub_c["copy_like_docstring_summary"], sub_c["has_leading_docstring_block"])

    print(f"Docstring-block rate FULL vs SUBSET: {full_doc_rate:.2f}% vs {sub_doc_rate:.2f}%")
    print(f"Copy-like rate (within docstring cases) FULL vs SUBSET: {full_copy_rate:.2f}% vs {sub_copy_rate:.2f}%")

    if sub_doc_rate > full_doc_rate + 2.0:
        print("→ SUBSET likely over-represents docstring code (can amplify bias).")
    if sub_copy_rate > full_copy_rate + 2.0:
        print("→ SUBSET likely over-represents summary≈docstring pairs (can amplify copying behavior).")


if __name__ == "__main__":
    main()
