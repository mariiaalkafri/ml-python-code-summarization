import json, os

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as g:
        for ex in rows:
            g.write(json.dumps(ex, ensure_ascii=False) + "\n")

def main():
    # CHANGE THESE ONLY IF YOUR FILENAMES ARE DIFFERENT
    train_orig_path = "data/processed/train.jsonl"
    valid_orig_path = "data/processed/valid.jsonl"

    train_clean_path = "data/processed/train_clean_v2.jsonl"
    valid_clean_path = "data/processed/valid_clean_v2.jsonl"

    train_mix_out = "data/processed/train_mix_v2.jsonl"
    valid_mix_out = "data/processed/valid_mix_v2.jsonl"

    train_orig = list(read_jsonl(train_orig_path))
    train_clean = list(read_jsonl(train_clean_path))
    valid_orig = list(read_jsonl(valid_orig_path))
    valid_clean = list(read_jsonl(valid_clean_path))

    train_mix = train_orig + train_clean
    valid_mix = valid_orig + valid_clean

    write_jsonl(train_mix_out, train_mix)
    write_jsonl(valid_mix_out, valid_mix)

    print(f"Saved {train_mix_out}: {len(train_mix)} (orig {len(train_orig)} + clean {len(train_clean)})")
    print(f"Saved {valid_mix_out}: {len(valid_mix)} (orig {len(valid_orig)} + clean {len(valid_clean)})")

if __name__ == "__main__":
    main()
