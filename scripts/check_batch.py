import sys
import os
import torch
from torch.utils.data import DataLoader

# Ensure we can import from src
sys.path.append(os.path.abspath("."))

from src.data import JsonlCodeSummaryDataset, Collator

def check_batch():
    tokenizer_path = "data/tokenizer/tokenizer.json"
    data_path = "data/processed/train.jsonl"
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    collator = Collator(tokenizer_path, max_src_len=512, max_tgt_len=128)
    
    print(f"Loading dataset from {data_path}...")
    dataset = JsonlCodeSummaryDataset(data_path)
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collator)
    
    print("Fetching one batch...")
    batch = next(iter(loader))
    
    print(f"Source IDs shape: {batch.src_ids.shape}")
    print(f"Source Mask shape: {batch.src_mask.shape}")
    print(f"Target IDs shape: {batch.tgt_ids.shape}")
    
    first_tgt = batch.tgt_ids[0].tolist()
    print(f"First target sequence (IDs): {first_tgt}")
    
    # Check BOS/EOS
    bos_id = collator.bos_id
    eos_id = collator.eos_id
    
    if first_tgt[0] == bos_id:
        print("PASS: First token is <bos>")
    else:
        print(f"FAIL: First token is {first_tgt[0]}, expected <bos> ({bos_id})")
        
    if eos_id in first_tgt:
        print("PASS: <eos> token found in sequence")
    else:
        print(f"FAIL: <eos> token NOT found in sequence (might be truncated if too long, or error)")

if __name__ == "__main__":
    check_batch()
