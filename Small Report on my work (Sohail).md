# Data Preparation and Tokenizer (My Part)

## Goal
Prepare a reliable training dataset for Python code summarization and provide a shared tokenizer so all group members use the same text-to-token conversion.

The model will learn from pairs of:
- code: a Python function/snippet
- summary: a short English description of what the code does

## Dataset Source
We use a public dataset from Hugging Face:
- google/code_x_glue_ct_code_to_text (Python split)

Each example contains:
- code (Python code)
- docstring (English documentation text)

We treat the docstring as the target summary to predict.

## Output Format (what the rest of the project uses)
The scripts generate local cleaned datasets in JSONL format:

- data/processed/train.jsonl
- data/processed/valid.jsonl
- data/processed/test.jsonl

Each line contains exactly:
- code: string
- summary: string

Example:
{"code":"def add(a,b): return a+b", "summary":"Return the sum of two numbers."}

## Cleaning / Preprocessing Steps
The goal of cleaning is to remove noisy or extreme examples and make the target summaries consistent.

Cleaning rules applied:
1) Remove empty examples
- discard rows with missing/empty code
- discard rows with missing/empty docstring/summary

2) Make summaries short and consistent
- keep only the first line / first sentence of the docstring as the summary
  This reduces noise (extra paragraphs, parameter lists, examples) and matches the goal of generating short summaries.

3) Remove very short summaries
- discard summaries shorter than a small threshold (to avoid useless labels)

4) Remove extremely long examples
- discard code snippets above a maximum length
- discard summaries above a maximum length
  This keeps training manageable and avoids outliers.

5) Remove duplicates
- if the exact same code appears multiple times, keep only one copy

## Cleaning Results
After cleaning, the script produced these counts:
- Train: kept 248,029, removed 3,791
- Valid: kept 13,663, removed 251
- Test: kept 14,609, removed 309

These numbers are stored in:
- data/processed/stats.json

## Tokenizer
A tokenizer is needed to convert text into token IDs before the model can train.

We trained a BPE tokenizer on the cleaned training split and saved it as:
- data/tokenizer/tokenizer.json

Tokenizer settings:
- model: BPE (Byte Pair Encoding)
- vocabulary size: 16,000
- special tokens: <pad>, <bos>, <eos>, <unk>

This tokenizer file is committed to the repository so all team members use the exact same tokenization and vocabulary.

## Embeddings Clarification
The embedding matrix is not created at the data-prep stage. It is part of the neural network model and will be created in the training code.
This part provides the tokenizer and vocabulary, which define the vocabulary size and special tokens needed to build the embedding layer.

## How to Reproduce (for teammates)
1) Create and activate .venv
2) Install dependencies: pip install -r requirements.txt
3) Generate cleaned dataset: python scripts/prepare_data.py
4) Tokenizer file used by everyone: data/tokenizer/tokenizer.json
