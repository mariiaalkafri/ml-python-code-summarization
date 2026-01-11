# scripts/train.py
import os
import sys
import random
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader

# Ensure we can import from src
sys.path.append(os.path.abspath("."))

from src.data import JsonlCodeSummaryDataset, Collator
from src.model import Seq2SeqLSTMAttn
from src.train_utils import train_model


# -----------------------------
# Subset selection (efficient, diverse)
# Stratify by:
#   - code length (chars)
#   - summary length (chars)
# Then sample across 2D buckets to ensure variety.
# -----------------------------
def _quantile_edges(lengths: List[int], buckets: int) -> List[int]:
    """
    Returns bucket edges using approximate quantiles.
    edges size = buckets-1
    """
    if not lengths:
        return []
    s = sorted(lengths)
    edges = []
    for i in range(1, buckets):
        # quantile index
        q = int(len(s) * i / buckets)
        q = min(max(q, 0), len(s) - 1)
        edges.append(s[q])
    # Ensure strictly non-decreasing edges (quantiles can repeat)
    return edges


def _bucket_id(x: int, edges: List[int]) -> int:
    """
    Put x into a bucket based on edges.
    buckets = len(edges)+1
    """
    for i, e in enumerate(edges):
        if x <= e:
            return i
    return len(edges)


def pick_2d_stratified_subset(
    examples: List[Dict],
    n: int,
    seed: int = 42,
    buckets_len: int = 5,
    buckets_sum: int = 3,
    verbose: bool = True,
) -> List[Dict]:
    """
    Select a diverse subset by stratifying across 2 dimensions:
      - code length buckets (quantiles)
      - summary length buckets (quantiles)

    Inside each (len_bucket, sum_bucket) cell:
      sample proportionally, but guarantees some coverage when possible.

    This is MUCH better than pure random because it forces variety:
      short/medium/long code and short/medium summaries.
    """
    if n is None or n <= 0 or n >= len(examples):
        if verbose:
            print("Subset: DISABLED (using full set)")
        return examples

    rng = random.Random(seed)

    code_lens = [len(ex.get("code", "")) for ex in examples]
    sum_lens = [len(ex.get("summary", "")) for ex in examples]

    len_edges = _quantile_edges(code_lens, buckets_len)
    sum_edges = _quantile_edges(sum_lens, buckets_sum)

    # Build 2D buckets
    cells: Dict[Tuple[int, int], List[Dict]] = {}
    for ex in examples:
        cl = len(ex.get("code", ""))
        sl = len(ex.get("summary", ""))
        bi = _bucket_id(cl, len_edges)  # 0..buckets_len-1
        bj = _bucket_id(sl, sum_edges)  # 0..buckets_sum-1
        cells.setdefault((bi, bj), []).append(ex)

    # How many cells actually have items?
    non_empty_cells = [k for k, v in cells.items() if len(v) > 0]
    if not non_empty_cells:
        # fallback random if something weird happens
        subset = rng.sample(examples, n)
        rng.shuffle(subset)
        return subset

    # Allocate quotas proportional to cell size
    total = len(examples)
    cell_items = [(k, len(cells[k])) for k in non_empty_cells]

    # base proportional quota
    quotas: Dict[Tuple[int, int], int] = {}
    for k, size in cell_items:
        q = int(round(n * (size / total)))
        quotas[k] = q

    # Fix rounding drift so sum quotas == n
    current = sum(quotas.values())

    # If we allocated too many, remove from largest quotas
    if current > n:
        extra = current - n
        for k, _ in sorted(cell_items, key=lambda x: quotas[x[0]], reverse=True):
            if extra <= 0:
                break
            if quotas[k] > 0:
                take = min(quotas[k], extra)
                quotas[k] -= take
                extra -= take

    # If we allocated too few, add to largest cells
    if current < n:
        missing = n - current
        for k, _ in sorted(cell_items, key=lambda x: x[1], reverse=True):
            if missing <= 0:
                break
            quotas[k] += 1
            missing -= 1

    # Guarantee coverage: if many cells exist, try to give 1 to each until we run out
    # (only if n is big enough)
    if n >= len(non_empty_cells):
        for k in non_empty_cells:
            if quotas[k] == 0:
                quotas[k] = 1
        # Re-normalize again if exceeded
        current = sum(quotas.values())
        if current > n:
            extra = current - n
            for k, _ in sorted(cell_items, key=lambda x: quotas[x[0]], reverse=True):
                if extra <= 0:
                    break
                if quotas[k] > 1:
                    take = min(quotas[k] - 1, extra)
                    quotas[k] -= take
                    extra -= take

    # Sample per cell
    subset: List[Dict] = []
    for k in non_empty_cells:
        q = quotas.get(k, 0)
        if q <= 0:
            continue
        bucket = cells[k]
        if q >= len(bucket):
            subset.extend(bucket)
        else:
            subset.extend(rng.sample(bucket, q))

    # If for any reason we got less than n (rare), top up randomly
    if len(subset) < n:
        remaining = [ex for ex in examples if ex not in subset]
        need = n - len(subset)
        if need > 0 and remaining:
            subset.extend(rng.sample(remaining, min(need, len(remaining))))

    rng.shuffle(subset)

    if verbose:
        print(f"Subset (2D stratified): selected {len(subset)} / {len(examples)}")
        # Print small bucket report (non-empty only)
        counts = []
        for (bi, bj), items in cells.items():
            if len(items) == 0:
                continue
            selected_here = min(quotas.get((bi, bj), 0), len(items))
            counts.append(((bi, bj), len(items), quotas.get((bi, bj), 0), selected_here))
        counts.sort(key=lambda x: (x[0][0], x[0][1]))
        print("Bucket report: (code_bucket, sum_bucket)  total_in_cell  quota")
        for (bi, bj), total_in_cell, q, _sel in counts:
            print(f"  ({bi},{bj})  total={total_in_cell}  quota={q}")

    return subset


def _maybe_mount_drive_if_needed(save_dir: str):
    """
    In Colab, if save_dir is under /content/drive, try to mount Drive automatically.
    If mount fails, we print what to do.
    """
    if not save_dir.startswith("/content/drive"):
        return

    try:
        import google.colab  # type: ignore
        from google.colab import drive  # type: ignore

        if not os.path.exists("/content/drive"):
            os.makedirs("/content/drive", exist_ok=True)

        print("Save dir is on Google Drive -> attempting to mount drive...")
        drive.mount("/content/drive", force_remount=False)
        print("Drive mounted.")
    except Exception as e:
        print("⚠️ Could not auto-mount Google Drive.")
        print("Run this in a separate Colab cell first:")
        print("from google.colab import drive\n"
              "drive.mount('/content/drive')")
        print("Error was:", str(e))


def main():
    # -----------------------------
    # PATHS
    # -----------------------------
    tokenizer_path = "data/tokenizer/tokenizer.json"
    train_path = "data/processed/train.jsonl"
    val_path = "data/processed/valid.jsonl"

    # -----------------------------
    # GOOD DEFAULTS FOR T4 (fast + stable)
    # -----------------------------
    batch_size = 32
    max_src_len = 256
    max_tgt_len = 64

    epochs = 10
    lr = 3e-4
    weight_decay = 0.01

    # subset settings (diverse, stratified)
    SUBSET_TRAIN = 50_000
    SUBSET_VAL = 5_000
    SEED = 42

    # Save on Drive (change folder name if you want)
    # If you don't want Drive, set save_dir="models"
    save_dir = "/content/drive/MyDrive/ml-python-code-summarization/models"

    # Model sizes (keep as you had; good baseline)
    emb_dim = 256
    enc_hidden = 256
    dec_hidden = 512
    num_layers = 1
    dropout = 0.2

    # DataLoader perf
    num_workers = 2
    pin_memory = True

    # Logging
    log_every = 200
    clip_grad = 1.0

    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Ensure Drive mounted if needed
    _maybe_mount_drive_if_needed(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print("Save dir:", save_dir)

    # Collator / tokenizer
    collator = Collator(tokenizer_path, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
    pad_id = collator.pad_id
    vocab_size = collator.tokenizer.get_vocab_size()

    # Load datasets
    print("Loading datasets...")
    train_dataset = JsonlCodeSummaryDataset(train_path)
    val_dataset = JsonlCodeSummaryDataset(val_path)

    print("Full train examples:", len(train_dataset))
    print("Full val examples:", len(val_dataset))

    # Apply stratified subset selection (NOT random-only)
    # We stratify train & val to keep them diverse too.
    train_dataset.examples = pick_2d_stratified_subset(
        train_dataset.examples,
        SUBSET_TRAIN,
        seed=SEED,
        buckets_len=5,
        buckets_sum=3,
        verbose=True,
    )
    val_dataset.examples = pick_2d_stratified_subset(
        val_dataset.examples,
        SUBSET_VAL,
        seed=SEED,
        buckets_len=5,
        buckets_sum=3,
        verbose=False,
    )

    print(f"Using subset sizes -> train={len(train_dataset)}  val={len(val_dataset)}")
    print(f"Hyperparams -> batch={batch_size} src_len={max_src_len} tgt_len={max_tgt_len} epochs={epochs} lr={lr}")

    # DataLoaders
    print("Building dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory and (device == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory and (device == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    print("Train batches/epoch:", len(train_loader))
    print("Val batches/epoch:", len(val_loader))

    # Model
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
        log_every=log_every,
        clip_grad=clip_grad,
        patience=3,
        min_delta=0.0,
        resume=True,  # resume from last.pt if exists
    )


if __name__ == "__main__":
    main()
