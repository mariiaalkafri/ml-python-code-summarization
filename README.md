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

## Team notes (what to do next)

I set up the repo structure (data/scripts/src) and added two scripts that handle the whole data pipeline.

First, the dataset: the script downloads a public Python dataset where each example already comes as a pair of Python code and an English description.

Then the cleaning: the script cleans the data so it’s consistent and easier to train on:
- removes empty or broken examples
- removes extremely long code/summary examples
- keeps only the first sentence of the description so summaries stay short and consistent
- removes duplicate code snippets

After cleaning, it creates three local files on your PC:
- data/processed/train.jsonl
- data/processed/valid.jsonl
- data/processed/test.jsonl
Each line in these files is: {"code": "...", "summary": "..."}.

I also trained and uploaded a shared tokenizer so we all use the exact same token splitting and token IDs:
- data/tokenizer/tokenizer.json

Now you guys have to do this on your own PC:
1) Clone the repo
2) Create and activate a virtual environment
3) Install the dependencies: pip install -r requirements.txt
4) Run the data script: python scripts/prepare_data.py
5) Use the tokenizer file that’s already in the repo (data/tokenizer/tokenizer.json) for training/evaluation

