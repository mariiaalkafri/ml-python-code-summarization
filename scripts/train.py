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
    Select a representative subset by stratifying by code length.
    Ensures short, medium, and long samples are included.
    """
    if n is None or n >= len(examples):
        return examples

    rng = random.Random(seed)

    # Sort by code length
    sorted_ex = sorted(examples, key=lambda x: len(x.get("code", "")))

    # Split into buckets
    bucket_lists = [sorted_ex[i::buckets] for i in range(buckets)]

    base = n // buckets
    remainder = n % buckets
    quotas = [base + (1 if i < remainder else 0) for i in range(buckets)]

    subset = []
    for bucket, q in zip(bucket_lists, quotas):
        if q >= len(bucket):
            subset.extend(bucket)
        else:
            subset.extend(rng.sample(bucket, q))

    rng.shuffle(subset)
    return subset


def main():
    # Paths
    tokenizer_path = "data/tokenizer/tokenizer.json"
    train_path = "data/processed/train.jsonl"
    val_path = "data/processed/valid.jsonl"

    # -----------------------------
    # TRAINING SETTINGS
    # -----------------------------
    batch_size = 16
    max_src_len = 256
    max_tgt_len = 64

    epochs = 10
    lr = 3e-4
    weight_decay = 0.01

    # 🔥 SUBSET SETTINGS (FINAL)
    SUBSET_TRAIN = 50_000
    SUBSET_VAL = 5_000
    SEED = 42
    # -----------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Collator / tokenizer
    collator = Collator(tokenizer_path, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
    pad_id = collator.pad_id
    vocab_size = collator.tokenizer.get_vocab_size()

    # Load datasets
    print("Loading datasets...")
    train_dataset = JsonlCodeSummaryDataset(train_path)
    val_dataset = JsonlCodeSummaryDataset(val_path)

    # Apply subset selection
    train_dataset.examples = pick_length_stratified_subset(
        train_dataset.examples, SUBSET_TRAIN, seed=SEED
    )
    val_dataset.examples = pick_length_stratified_subset(
        val_dataset.examples, SUBSET_VAL, seed=SEED
    )

    print(f"Using subset: train={len(train_dataset)}  val={len(val_dataset)}")

    # DataLoaders
    print("Building dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True
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
