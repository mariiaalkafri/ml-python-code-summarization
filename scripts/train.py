# scripts/train.py
import os
import sys
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Ensure we can import from src/
sys.path.append(os.path.abspath("."))

from src.data import JsonlCodeSummaryDataset, Collator
from src.model import Seq2SeqLSTMAttn
from src.train_utils import train_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_subset(dataset, subset_size: int, seed: int):
    """
    Random subset (reproducible).
    Prints exactly how many samples you train on.
    """
    n = len(dataset)
    if subset_size <= 0 or subset_size >= n:
        return dataset, None  # no subset

    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=subset_size, replace=False)
    indices = indices.tolist()
    return Subset(dataset, indices), indices


def main():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer/tokenizer.json")
    parser.add_argument("--train_path", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--val_path", type=str, default="data/processed/valid.jsonl")

    # Subset selection
    parser.add_argument("--subset_size", type=int, default=50000,
                        help="How many training samples to use. Set 0 to disable.")
    parser.add_argument("--subset_seed", type=int, default=42,
                        help="Seed for subset selection (reproducible).")

    # Speed knobs
    parser.add_argument("--max_src_len", type=int, default=256)
    parser.add_argument("--max_tgt_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)

    # Training knobs
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--clip_grad", type=float, default=1.0)

    # Early stopping
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.0)

    # Model knobs
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--enc_hidden", type=int, default=256)
    parser.add_argument("--dec_hidden", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Logging & checkpoints
    parser.add_argument("--save_dir", type=str, default="models",
                        help="Folder where best.pt and last.pt will be stored.")
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from save_dir/last.pt if it exists.")

    args = parser.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading datasets...")
    print("Train file:", args.train_path)
    print("Val file:", args.val_path)

    # Tokenizer + collator
    collator = Collator(args.tokenizer_path, max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
    vocab_size = collator.tokenizer.get_vocab_size()
    pad_id = collator.pad_id

    # Load datasets
    full_train_dataset = JsonlCodeSummaryDataset(args.train_path)
    val_dataset = JsonlCodeSummaryDataset(args.val_path)

    print("Full train examples:", len(full_train_dataset))
    print("Val examples:", len(val_dataset))

    # Subset selection (TRAIN only)
    train_dataset, subset_indices = make_subset(full_train_dataset, args.subset_size, args.subset_seed)

    if subset_indices is None:
        print("Subset: DISABLED (using full train set)")
        train_examples = len(full_train_dataset)
    else:
        print(f"Subset: ENABLED -> using {len(train_dataset)} samples out of {len(full_train_dataset)}")
        print(f"Subset seed: {args.subset_seed}")
        train_examples = len(train_dataset)

    print("Building dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    print("Batch size:", args.batch_size)
    print("Train batches per epoch:", len(train_loader))
    print("Val batches per epoch:", len(val_loader))

    # Model
    print("Initializing model...")
    model = Seq2SeqLSTMAttn(
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        enc_hidden=args.enc_hidden,
        dec_hidden=args.dec_hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_id=pad_id,
    ).to(device)

    # Save dir
    os.makedirs(args.save_dir, exist_ok=True)
    print("Save dir:", args.save_dir)
    print("Checkpoints will be:")
    print("  Best:", os.path.join(args.save_dir, "best.pt"))
    print("Resume:", "ON" if args.resume else "OFF")

    # Train
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        pad_id=pad_id,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_grad=args.clip_grad,
        save_dir=args.save_dir,
        log_every=args.log_every,
        patience=args.patience,
        min_delta=args.min_delta,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
