import json
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from dataclasses import dataclass
from typing import List, Dict, Any
import os

@dataclass
class Batch:
    src_ids: torch.Tensor
    src_mask: torch.Tensor
    tgt_ids: torch.Tensor

class JsonlCodeSummaryDataset(Dataset):
    def __init__(self, file_path: str):
        self.examples = []
        if not os.path.exists(file_path):
             # Just handle empty case gracefully or raise error, strictly following instruction to assume files exist/handle logic
             pass
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

class Collator:
    def __init__(self, tokenizer_path: str, max_src_len: int = 512, max_tgt_len: int = 128):
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
            
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # Check special tokens
        pad_id = self.tokenizer.token_to_id("<pad>")
        bos_id = self.tokenizer.token_to_id("<bos>")
        eos_id = self.tokenizer.token_to_id("<eos>")
        unk_id = self.tokenizer.token_to_id("<unk>")

        if any(x is None for x in [pad_id, bos_id, eos_id, unk_id]):
            raise ValueError("Tokenizer must have <pad>, <bos>, <eos>, <unk> tokens")

        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Batch:
        src_batch = []
        tgt_batch = []

        for item in batch:
            src_batch.append(item['code'])
            tgt_batch.append(item['summary'])

        # Encode sources
        src_encodings = self.tokenizer.encode_batch(src_batch)
        
        # Prepare source tensors
        src_ids_list = []
        src_masks_list = []

        for enc in src_encodings:
            ids = enc.ids
            # Truncate
            if len(ids) > self.max_src_len:
                ids = ids[:self.max_src_len]
            
            length = len(ids)
            # Create mask (1 for valid, 0 for pad)
            mask = [1] * length
            
            # Pad
            if length < self.max_src_len:
                pad_len = self.max_src_len - length
                ids = ids + [self.pad_id] * pad_len
                mask = mask + [0] * pad_len
            
            src_ids_list.append(ids)
            src_masks_list.append(mask)

        src_ids = torch.tensor(src_ids_list, dtype=torch.long)
        src_mask = torch.tensor(src_masks_list, dtype=torch.long)

        # Encode targets with special tokens <bos> ... <eos>
        tgt_ids_list = []
        
        # We process targets manually to ensure BOS/EOS placement and manual truncation/padding
        tgt_encodings = self.tokenizer.encode_batch(tgt_batch)

        for enc in tgt_encodings:
            ids = enc.ids
            # Truncate to max_tgt_len - 2 (to fit BOS and EOS)
            if len(ids) > self.max_tgt_len - 2:
                ids = ids[:self.max_tgt_len - 2]
            
            # Add BOS and EOS
            ids = [self.bos_id] + ids + [self.eos_id]
            
            length = len(ids)
            # Pad
            if length < self.max_tgt_len:
                pad_len = self.max_tgt_len - length
                ids = ids + [self.pad_id] * pad_len
            
            tgt_ids_list.append(ids)

        tgt_ids = torch.tensor(tgt_ids_list, dtype=torch.long)

        return Batch(src_ids=src_ids, src_mask=src_mask, tgt_ids=tgt_ids)
