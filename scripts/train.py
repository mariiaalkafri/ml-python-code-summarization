import sys
import os
import random
import torch
from torch.utils.data import DataLoader
from typing import List, Dict

# Ensure we can import from src
sys.path.append(os.path.abspath("."))

from src.data import JsonlCodeSummaryDataset, Collator
from src.model import Seq2SeqLSTMAttn
from src.train_utils import train_model


def pick_length_stratified_subset(examples: List[Dict], n: int, seed: int = 42, buckets: int = 5):
    """
    Picks a representative subset by stratifying examples by code length.
    This avoids bias from taking the first N rows and ensures we keep short/medium/long code.

    - examples: list of {"code":..., "summary":...}
    - n: subset size (if n >= len(examples), returns full list)
    - seed: reproducible sampling
    - buckets: number of length buckets (default 5)
    """
    if n is None or n <= 0 or n >= len(examples):
        return examples

    rng = random.Random(seed)

    # Sort by code length
    sorted_ex = sorted(examples, key=lambda x: len(x.get("code", "")))

    # Split into buckets by interleaving to keep buckets balanced
    b = max(2, int(buckets))
    bucket_lists = [sorted_ex[i::b] for i in range(b)]

    base = n // b
    extra = n % b
    quotas = [base + (1 if i < extra else 0) for i in range(b)]

    picked = []
    for bucket, q in zip(bucket_lists, quotas):
        if not bucket:
            continue
        if q >= len(bucket):
            picked.extend(bucket)
        else:
            picked.extend(rng.sample(bucket, q))

    rng.shuffle(picked)
    return picked


def main():
    # Paths
    tokenizer_path = "data/tokenizer/tokenizer.json"
    train_path = "data/processed/train.jsonl"
    val_path = "data/processed/valid.jsonl"

    # -----------------------------
    # SPEED / QUALITY SETTINGS
    # -----------------------------
    # Sequence lengths (major speed lever for attention models)
    batch_size = 16
    max_src_len = 256
    max_tgt_len = 64

    # Training schedule
    epochs = 10
    lr = 3e-4
    weight_decay = 0.01

    # SUBSET SETTINGS (set to None for full training)
    # Recommended starting point:
    SUBSET_TRAIN = 100_000   # try 50_000 / 100_000 / 150_000
    SUBSET_VAL   = 10_000
    SEED = 42
    # -----------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data + Tokenizer (through collator)
    collator = Collator(tokenizer_path, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
    vocab_size = collator.tokenizer.get_vocab_size()
    pad_id = collator.pad_id

    print("Loading datasets...")
    train_dataset = JsonlCodeSummaryDataset(train_path)
    val_dataset = JsonlCodeSummaryDataset(val_path)

    # Apply subset selection (ONLY affects training time, not correctness)
    # Dataset uses `.examples` in your implementation.
    if hasattr(train_dataset, "examples") and isinstance(train_dataset.examples, list):
        train_dataset.examples = pick_length_stratified_subset(train_dataset.examples, SUBSET_TRAIN, seed=SEED)
    if hasattr(val_dataset, "examples") and isinstance(val_dataset.examples, list):
        val_dataset.examples = pick_length_stratified_subset(val_dataset.examples, SUBSET_VAL, seed=SEED)

    print(f"Using subset sizes: train={len(train_dataset)}  val={len(val_dataset)}")

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

    # Model
    print("Initializing model...")
    model = Seq2SeqLSTMAttn(
        vocab_size=vocab_size,
        emb_dim=256,
        enc_hidden=256,
        dec_hidden=512,
        num_layers=1,
        dropout=0.2,
        pad_id=pad_id
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
        save_dir="models",
        log_every=200,
        clip_grad=1.0
    )


if __name__ == "__main__":
    main()
