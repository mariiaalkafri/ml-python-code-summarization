import sys
import os
import random
from typing import List, Dict, Tuple

import torch
from torch.utils.data import DataLoader

# Ensure we can import from src (repo root)
sys.path.append(os.path.abspath("."))

from src.data import JsonlCodeSummaryDataset, Collator
from src.transformer_model import TransformerSeq2Seq
from src.train_utils_transformer import train_transformer_model

def _bucket_index(value: int, thresholds: List[int]) -> int:
    for i, t in enumerate(thresholds):
        if value <= t:
            return i
    return len(thresholds)

def pick_2d_stratified_subset(
    examples: List[Dict],
    n: int,
    seed: int = 42,
    code_thresholds: List[int] = None,
    sum_thresholds: List[int] = None,
) -> Tuple[List[Dict], List[Tuple[Tuple[int, int], int, int]]]:
    if n is None or n >= len(examples):
        return examples, []

    rng = random.Random(seed)

    if code_thresholds is None:
        code_thresholds = [200, 600, 1200, 2000]
    if sum_thresholds is None:
        sum_thresholds = [60, 120]

    buckets = {}
    for ex in examples:
        code_len = len(ex.get("code", ""))
        sum_len = len(ex.get("summary", ""))
        cb = _bucket_index(code_len, code_thresholds)
        sb = _bucket_index(sum_len, sum_thresholds)
        key = (cb, sb)
        buckets.setdefault(key, []).append(ex)

    total = len(examples)
    ideal = {}
    for key, bucket_list in buckets.items():
        ideal[key] = n * (len(bucket_list) / total)

    quotas = {k: int(v) for k, v in ideal.items()}
    taken = sum(quotas.values())

    remainder = n - taken
    if remainder > 0:
        fracs = sorted(
            [(k, ideal[k] - quotas[k]) for k in quotas.keys()],
            key=lambda x: x[1],
            reverse=True,
        )
        for i in range(remainder):
            quotas[fracs[i % len(fracs)][0]] += 1

    subset = []
    bucket_report = []
    for key in sorted(buckets.keys()):
        bucket_list = buckets[key]
        q = quotas.get(key, 0)
        bucket_report.append((key, len(bucket_list), q))

        if q <= 0:
            continue
        if q >= len(bucket_list):
            subset.extend(bucket_list)
        else:
            subset.extend(rng.sample(bucket_list, q))

    rng.shuffle(subset)
    return subset, bucket_report



def main():
    tokenizer_path = "data/tokenizer/tokenizer.json"
    train_path = "data/processed/train.jsonl"
    val_path = "data/processed/valid.jsonl"

    batch_size = 32
    max_src_len = 256
    max_tgt_len = 64

    epochs = 10
    lr = 3e-4
    weight_decay = 0.01

    SUBSET_TRAIN = 50_000
    SUBSET_VAL = 5_000
    SEED = 42

    save_dir = "/content/drive/MyDrive/ml-python-code-summarization/models_transformer" # Changed dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(save_dir, exist_ok=True)
    print(f"Save dir: {save_dir}")

    collator = Collator(tokenizer_path, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
    pad_id = collator.pad_id
    vocab_size = collator.tokenizer.get_vocab_size()

    print("Loading datasets...")
    train_dataset = JsonlCodeSummaryDataset(train_path)
    val_dataset = JsonlCodeSummaryDataset(val_path)

    print(f"Full train examples: {len(train_dataset)}")
    print(f"Full val examples: {len(val_dataset)}")

    train_subset, train_report = pick_2d_stratified_subset(
        train_dataset.examples, SUBSET_TRAIN, seed=SEED
    )
    val_subset, _ = pick_2d_stratified_subset(
        val_dataset.examples, SUBSET_VAL, seed=SEED
    )

    train_dataset.examples = train_subset
    val_dataset.examples = val_subset

    print(f"Subset (2D stratified): selected {len(train_dataset)}")
    if train_report:
        print("Bucket report: (code_bucket, sum_bucket)  total_in_cell  quota")
        for (cb, sb), total_in_cell, quota in train_report:
            print(f"  ({cb},{sb})  total={total_in_cell}  quota={quota}")

    print(f"Using subset sizes -> train={len(train_dataset)}  val={len(val_dataset)}")
    print(f"Hyperparams -> batch={batch_size} src_len={max_src_len} tgt_len={max_tgt_len} epochs={epochs} lr={lr}")

    print("Building dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Train batches/epoch: {len(train_loader)}")
    print(f"Val batches/epoch: {len(val_loader)}")

    print("Initializing Transformer model...")
    model = TransformerSeq2Seq(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        pad_id=pad_id,
    ).to(device)

    print("Starting training...")
    
    train_transformer_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        pad_id=pad_id,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        save_dir=save_dir,
        log_every=200,
        clip_grad=1.0,
    )

if __name__ == "__main__":
    main()
