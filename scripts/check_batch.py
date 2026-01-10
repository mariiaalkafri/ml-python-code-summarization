import os
import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from src.data import JsonlCodeSummaryDataset, Collator

def main():
    tokenizer_path = "data/tokenizer/tokenizer.json"
    train_path = "data/processed/train.jsonl"

    print(f"Loading tokenizer from {tokenizer_path}...")
    tok = Tokenizer.from_file(tokenizer_path)

    print(f"Loading dataset from {train_path}...")
    ds = JsonlCodeSummaryDataset(train_path)

    # Match your training settings here
    max_src_len = 256
    max_tgt_len = 64
    batch_size = 4

    collator = Collator(tokenizer_path, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collator)

    print("Fetching one batch...")
    batch = next(iter(loader))

    print("Source IDs shape:", batch.src_ids.shape)
    print("Source Mask shape:", batch.src_mask.shape)
    print("Target IDs shape:", batch.tgt_ids.shape)

    # Inspect special tokens
    bos_id = collator.bos_id
    eos_id = collator.eos_id

    first = batch.tgt_ids[0].tolist()
    print("First target sequence (IDs):", first)

    if first[0] == bos_id:
        print("PASS: First token is <bos>")
    else:
        print("FAIL: First token is NOT <bos>")

    if eos_id in first:
        print("PASS: <eos> token found in sequence")
    else:
        print("FAIL: <eos> token NOT found")

if __name__ == "__main__":
    main()
