# scripts/train.py
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
    # -------------------------
    # Configuration
    # -------------------------
    tokenizer_path = "data/tokenizer/tokenizer.json"
    train_path = "data/processed/train.jsonl"
    val_path = "data/processed/valid.jsonl"

    # -------------------------
    # Hyperparameters (speed-friendly)
    # -------------------------
    batch_size = 16
    max_src_len = 256   # was 512 (huge speed-up)
    max_tgt_len = 64    # was 128 (huge speed-up)

    epochs = 10
    lr = 3e-4
    weight_decay = 0.01
    clip_grad = 1.0
    log_every = 200

    # Model size
    emb_dim = 256
    enc_hidden = 256
    dec_hidden = 512
    num_layers = 1
    dropout = 0.2

    # -------------------------
    # Device
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -------------------------
    # Data
    # -------------------------
    print("Loading datasets...")
    collator = Collator(tokenizer_path, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
    vocab_size = collator.tokenizer.get_vocab_size()
    pad_id = collator.pad_id

    train_dataset = JsonlCodeSummaryDataset(train_path)
    val_dataset = JsonlCodeSummaryDataset(val_path)

    print("Building dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # -------------------------
    # Model
    # -------------------------
    print("Initializing model...")
    model = Seq2SeqLSTMAttn(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        enc_hidden=enc_hidden,
        dec_hidden=dec_hidden,
        num_layers=num_layers,
        dropout=dropout,
        pad_id=pad_id
    ).to(device)

    # -------------------------
    # Save directory
    # -------------------------
    # If you mount Drive on Colab, change to:
    # save_dir = "/content/drive/MyDrive/mlsa_models"
    save_dir = "models"

    # -------------------------
    # Train
    # -------------------------
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
        clip_grad=clip_grad,
        log_every=log_every,
        save_dir=save_dir,
        resume=True
    )


if __name__ == "__main__":
    main()
