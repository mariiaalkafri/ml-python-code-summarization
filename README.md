# ML-Based Python Code Summarization

## Setup (Windows / PowerShell)

Run:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Data preparation (downloads + cleans)

This produces:
- data/processed/train.jsonl
- data/processed/valid.jsonl
- data/processed/test.jsonl
- data/processed/stats.json

Run:

    python scripts\prepare_data.py

## Tokenizer training

This produces:
- data/tokenizer/tokenizer.json

Run:

    python scripts\train_tokenizer.py
