# scripts/train_tokenizer.py
import os, json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

TRAIN_FILE = "data/processed/train.jsonl"
OUT_DIR = "data/tokenizer"
os.makedirs(OUT_DIR, exist_ok=True)

# special tokens the model will need
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

# choose vocabulary size (how many token types the tokenizer will learn)
VOCAB_SIZE = 16000

def iter_text():
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            yield x["code"]
            yield x["summary"]

def main():
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
    tok.train_from_iterator(iter_text(), trainer=trainer)

    out_path = os.path.join(OUT_DIR, "tokenizer.json")
    tok.save(out_path)
    print("Saved tokenizer to:", out_path)

if __name__ == "__main__":
    main()
