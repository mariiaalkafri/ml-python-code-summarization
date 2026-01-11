import os
import sys
import json
import random
import torch

sys.path.append(os.path.abspath("."))

from src.inference import load_inference_model, summarize_code


def find_model_path():
    if os.path.exists("models/best.pt"):
        return "models/best.pt"
    drive_path = "/content/drive/MyDrive/ml-python-code-summarization/models/best.pt"
    if os.path.exists(drive_path):
        return drive_path
    return "models/best.pt"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    test_path = "data/processed/test.jsonl"
    tokenizer_path = "data/tokenizer/tokenizer.json"
    model_path = find_model_path()

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test set: {test_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Test: {test_path}")

    # ---- evaluation config ----
    N = 200          # start small (200). later try 1000+
    SEED = 42
    MAX_SRC_LEN = 256
    MAX_GEN_LEN = 64
    # --------------------------

    # install / import rouge metric
    # (works in Colab; locally install evaluate via pip)
    import evaluate
    rouge = evaluate.load("rouge")

    # load model/tokenizer
    model, tokenizer = load_inference_model(model_path, tokenizer_path, device)

    # load data
    data = load_jsonl(test_path)
    print(f"Test examples available: {len(data)}")

    rng = random.Random(SEED)
    if N < len(data):
        sample = rng.sample(data, N)
    else:
        sample = data

    preds = []
    refs = []

    for i, ex in enumerate(sample, 1):
        code = ex["code"]
        ref = ex["summary"]

        pred = summarize_code(
            model=model,
            tokenizer=tokenizer,
            code=code,
            device=device,
            max_src_len=MAX_SRC_LEN,
            max_len=MAX_GEN_LEN,
        )

        preds.append(pred)
        refs.append(ref)

        if i <= 5:
            print("\n--- Example", i, "---")
            print("REF :", ref)
            print("PRED:", pred)

        if i % 50 == 0:
            print(f"Processed {i}/{len(sample)}")

    results = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    print("\nROUGE results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()

