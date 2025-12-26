# scripts/prepare_data.py
import os, json, re
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Simple cleaning limits (characters)
MAX_CODE_CHARS = 4000
MAX_SUMMARY_CHARS = 400
MIN_SUMMARY_CHARS = 10

def first_sentence(text: str) -> str:
    """Keep the first line / first sentence of the docstring to make summaries short."""
    text = (text or "").strip()
    if not text:
        return ""
    # First line only (simple + safe)
    text = text.splitlines()[0].strip()
    # If the first line contains multiple sentences, keep only the first sentence
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return parts[0].strip()

def clean_pair(code: str, docstring: str):
    code = (code or "").strip()
    summary = first_sentence(docstring)

    if not code or not summary:
        return None
    if len(summary) < MIN_SUMMARY_CHARS:
        return None
    if len(code) > MAX_CODE_CHARS:
        return None
    if len(summary) > MAX_SUMMARY_CHARS:
        return None

    return {"code": code, "summary": summary}

def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    # CodeXGLUE Code-to-Text (Python): has fields "code" and "docstring"
    ds = load_dataset("google/code_x_glue_ct_code_to_text", "python")

    split_map = {"train": "train", "validation": "valid", "test": "test"}
    stats = {}

    for hf_split, out_name in split_map.items():
        kept = []
        removed = 0
        seen_code = set()  # remove exact duplicates by code text

        for x in tqdm(ds[hf_split], desc=f"Processing {hf_split}"):
            r = clean_pair(x.get("code"), x.get("docstring"))
            if r is None:
                removed += 1
                continue
            if r["code"] in seen_code:
                continue
            seen_code.add(r["code"])
            kept.append(r)

        out_path = os.path.join(OUT_DIR, f"{out_name}.jsonl")
        write_jsonl(out_path, kept)

        stats[out_name] = {"kept": len(kept), "removed": removed}
        print(f"{out_name}: kept={len(kept)} removed={removed}")

    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Saved cleaned splits to data/processed/")

if __name__ == "__main__":
    main()
