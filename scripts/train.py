import sys
import os
import torch
from torch.utils.data import DataLoader

# Ensure we can import from src
sys.path.append(os.path.abspath("."))

from src.data import JsonlCodeSummaryDataset, Collator
from src.model import Seq2SeqLSTMAttn
from src.train_utils import train_model


def main():
    # Paths
    tokenizer_path = "data/tokenizer/tokenizer.json"
    train_path = "data/processed/train.jsonl"
    val_path = "data/processed/valid.jsonl"

    # Hyperparameters (speed-friendly defaults for Colab T4)
    batch_size = 16
    max_src_len = 256   # was 512 -> much faster
    max_tgt_len = 64    # was 128 -> much faster
    epochs = 10         # early stopping will stop earlier if needed
    lr = 3e-4
    weight_decay = 0.01

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Data
    collator = Collator(tokenizer_path, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
    vocab_size = collator.tokenizer.get_vocab_size()
    pad_id = collator.pad_id

    print("Loading datasets...")
    train_dataset = JsonlCodeSummaryDataset(train_path)
    val_dataset = JsonlCodeSummaryDataset(val_path)

    # Optional smoke-test subset (uncomment if you want a fast test first)
    # train_dataset.examples = train_dataset.examples[:2000]
    # val_dataset.examples = val_dataset.examples[:500]

    # DataLoaders (GPU-friendly)
    # Note: persistent_workers may fail in some Colab sessions; we handle it safely.
    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    try:
        train_loader = DataLoader(train_dataset, shuffle=True, persistent_workers=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, persistent_workers=True, **loader_kwargs)
    except TypeError:
        # Fallback if persistent_workers isn't supported
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

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
        save_dir="models"
    )


if __name__ == "__main__":
    main()
