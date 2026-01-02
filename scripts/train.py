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
    # Configuration
    tokenizer_path = "data/tokenizer/tokenizer.json"
    train_path = "data/processed/train.jsonl"
    val_path = "data/processed/valid.jsonl"
    
    # Hyperparameters
    batch_size = 16 
    max_src_len = 512
    max_tgt_len = 128
    epochs = 20
    lr = 3e-4
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Data
    collator = Collator(tokenizer_path, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
    vocab_size = collator.tokenizer.get_vocab_size()
    pad_id = collator.pad_id
    
    print("Loading datasets...")
    train_dataset = JsonlCodeSummaryDataset(train_path)
    val_dataset = JsonlCodeSummaryDataset(val_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    
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
        vocab_size=vocab_size,
        pad_id=pad_id,
        epochs=epochs,
        lr=lr
    )

if __name__ == "__main__":
    main()
