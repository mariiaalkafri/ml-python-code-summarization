import sys
import os
import random
from typing import List, Dict, Tuple

import torch
from torch.utils.data import DataLoader

# Ensure we can import from src (repo root)
sys.path.append(os.path.abspath("."))

from src.data import JsonlCodeSummaryDataset, Collator
from src.model import Seq2SeqLSTMAttn
from src.train_utils import train_model


def _bucket_index(value: int, thresholds: List[int]) -> int:
    """
    thresholds are upper bounds for buckets except last bucket.
    Example thresholds [200, 600, 1200, 2000] -> 5 buckets:
      0: <=200
      1: 201-600
      2: 601-1200
      3: 1201-2000
      4: >2000
    """
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
    """
    Select a representative subset stratified by BOTH:
      - code length (characters)
      - summary length (characters)

    This avoids "pure random" sampling and ensures the model sees
    short/medium/long code and short/medium/long summaries.

    Returns:
      subset_examples
      bucket_report: list of ((code_bucket, sum_bucket), total_in_cell, quota)
    """
    if n is None or n >= len(examples):
        # no subsetting needed
        return examples, []

    rng = random.Random(seed)

    if code_thresholds is None:
        code_thresholds = [200, 600, 1200, 2000]  # 5 buckets
    if sum_thresholds is None:
        sum_thresholds = [60, 120]  # 3 buckets

    # 1) group examples into 2D buckets
    buckets = {}  # (cb, sb) -> list
    for ex in examples:
        code_len = len(ex.get("code", ""))
        sum_len = len(ex.get("summary", ""))
        cb = _bucket_index(code_len, code_thresholds)
        sb = _bucket_index(sum_len, sum_thresholds)
        key = (cb, sb)
        buckets.setdefault(key, []).append(ex)

    # 2) allocate quotas proportional to bucket sizes
    total = len(examples)
    # compute ideal quota (float)
    ideal = {}
    for key, bucket_list in buckets.items():
        ideal[key] = n * (len(bucket_list) / total)

    # take floor first
    quotas = {k: int(v) for k, v in ideal.items()}
    taken = sum(quotas.values())

    # distribute remainder by largest fractional parts
    remainder = n - taken
    if remainder > 0:
        fracs = sorted(
            [(k, ideal[k] - quotas[k]) for k in quotas.keys()],
            key=lambda x: x[1],
            reverse=True,
        )
        for i in range(remainder):
            quotas[fracs[i % len(fracs)][0]] += 1

    # 3) sample from each bucket
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
    # -----------------------------
    # PATHS
    # -----------------------------
    tokenizer_path = "data/tokenizer/tokenizer.json"
    train_path = "data/processed/train.jsonl"
    val_path = "data/processed/valid.jsonl"

    # -----------------------------
    # TRAINING SETTINGS
    # -----------------------------
    batch_size = 32
    max_src_len = 256
    max_tgt_len = 64

    epochs = 10
    lr = 3e-4
    weight_decay = 0.01

    # -----------------------------
    # SUBSET SETTINGS
    # -----------------------------
    SUBSET_TRAIN = 50_000
    SUBSET_VAL = 5_000
    SEED = 42

    # -----------------------------
    # SAVE DIR (GOOGLE DRIVE)
    # IMPORTANT: do NOT mount drive inside this script.
    # Mount in a Colab cell first:
    # from google.colab import drive
    # drive.mount('/content/drive')
    # -----------------------------
    save_dir = "/content/drive/MyDrive/ml-python-code-summarization/models"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Ensure save dir exists
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save dir: {save_dir}")

    # Collator / tokenizer
    collator = Collator(tokenizer_path, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
    pad_id = collator.pad_id
    vocab_size = collator.tokenizer.get_vocab_size()

    # Load full datasets
    print("Loading datasets...")
    train_dataset = JsonlCodeSummaryDataset(train_path)
    val_dataset = JsonlCodeSummaryDataset(val_path)

    print(f"Full train examples: {len(train_dataset)}")
    print(f"Full val examples: {len(val_dataset)}")

    # Apply stratified subset (not random)
    train_subset, train_report = pick_2d_stratified_subset(
        train_dataset.examples, SUBSET_TRAIN, seed=SEED
    )
    val_subset, _ = pick_2d_stratified_subset(
        val_dataset.examples, SUBSET_VAL, seed=SEED
    )

    train_dataset.examples = train_subset
    val_dataset.examples = val_subset

    print(f"Subset (2D stratified): selected {len(train_dataset)} / {248029 if len(train_subset)!=0 else len(train_dataset)}")
    if train_report:
        print("Bucket report: (code_bucket, sum_bucket)  total_in_cell  quota")
        for (cb, sb), total_in_cell, quota in train_report:
            print(f"  ({cb},{sb})  total={total_in_cell}  quota={quota}")

    print(f"Using subset sizes -> train={len(train_dataset)}  val={len(val_dataset)}")
    print(f"Hyperparams -> batch={batch_size} src_len={max_src_len} tgt_len={max_tgt_len} epochs={epochs} lr={lr}")

    # Dataloaders
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

    # Model
    print("Initializing model...")
    model = Seq2SeqLSTMAttn(
        vocab_size=vocab_size,
        emb_dim=256,
        enc_hidden=256,
        dec_hidden=512,
        num_layers=1,
        dropout=0.2,
        pad_id=pad_id,
    ).to(device)

    # Train
    print("Starting training...")
    train_model(
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
