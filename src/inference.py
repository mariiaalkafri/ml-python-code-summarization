import os
import sys
import torch

sys.path.append(os.path.abspath("."))

from src.inference import load_inference_model, summarize_code


def main():
    # ✅ CHANGE THIS to your real Drive pathimport os
from typing import Tuple, List

import torch
from tokenizers import Tokenizer

from src.model import Seq2SeqLSTMAttn


def load_inference_model(
    model_path: str,
    tokenizer_path: str,
    device: str,
    emb_dim: int = 256,
    enc_hidden: int = 256,
    dec_hidden: int = 512,
    num_layers: int = 1,
) -> Tuple[Seq2SeqLSTMAttn, Tokenizer]:
    """
    Loads the trained Seq2Seq model + tokenizer for inference.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if pad_id is None or bos_id is None or eos_id is None:
        raise ValueError("Tokenizer must contain <pad>, <bos>, <eos> tokens")

    model = Seq2SeqLSTMAttn(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        enc_hidden=enc_hidden,
        dec_hidden=dec_hidden,
        num_layers=num_layers,
        dropout=0.0,  # eval
        pad_id=pad_id,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    return model, tokenizer


def _strip_special_ids(ids: List[int], pad_id: int, bos_id: int, eos_id: int) -> List[int]:
    """
    Remove special tokens and stop at EOS.
    Works even if tokenizer.json does NOT mark them as special.
    """
    cleaned = []
    for t in ids:
        if t == eos_id:
            break
        if t in (pad_id, bos_id):
            continue
        cleaned.append(t)
    return cleaned


def _collapse_repetitions(tokens: List[int], max_repeat: int = 3) -> List[int]:
    """
    Very small safety: prevents ugly loops like "b b b b b b ..."
    by limiting consecutive repeats.
    """
    if not tokens:
        return tokens

    out = [tokens[0]]
    run = 1
    for t in tokens[1:]:
        if t == out[-1]:
            run += 1
            if run <= max_repeat:
                out.append(t)
        else:
            run = 1
            out.append(t)
    return out


@torch.no_grad()
def summarize_code(
    model: Seq2SeqLSTMAttn,
    tokenizer: Tokenizer,
    code: str,
    device: str,
    max_src_len: int = 256,
    max_gen_len: int = 64,
) -> str:
    """
    Greedy generation using model.generate(), with correct truncation/mask
    and robust decoding (manual special-token stripping).
    """

    code = (code or "").strip()
    if not code:
        return ""

    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    enc = tokenizer.encode(code)
    src = enc.ids[:max_src_len]

    # pad to max_src_len so mask matches training style
    if len(src) < max_src_len:
        src = src + [pad_id] * (max_src_len - len(src))

    src_ids = torch.tensor([src], dtype=torch.long, device=device)
    src_mask = (src_ids != pad_id).long()  # 1 where real, 0 where pad

    # Generate (assumes your model has .generate implemented)
    gen_ids = model.generate(
        src_ids=src_ids,
        src_mask=src_mask,
        max_len=max_gen_len,
        bos_id=bos_id,
        eos_id=eos_id,
    )

    # gen_ids could be tensor or list
    if isinstance(gen_ids, torch.Tensor):
        gen_ids = gen_ids.squeeze(0).tolist()

    # Clean output ids
    gen_ids = _strip_special_ids(gen_ids, pad_id, bos_id, eos_id)
    gen_ids = _collapse_repetitions(gen_ids, max_repeat=3)

    if not gen_ids:
        return ""

    # decode WITHOUT relying on "special token" metadata
    summary = tokenizer.decode(gen_ids)
    return summary.strip()

    model_path = "/content/drive/MyDrive/ml-python-code-summarization/models/best.pt"
    tokenizer_path = "data/tokenizer/tokenizer.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Model: {model_path}\n")

    model, tokenizer = load_inference_model(model_path, tokenizer_path, device)

    print("✅ Model loaded. Paste code. Finish with an empty line. Ctrl+C to exit.")
    print("-" * 60)

    while True:
        try:
            print(">> Enter Code (end with empty line):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)

            code = "\n".join(lines).strip()
            if not code:
                continue

            summary = summarize_code(
                model=model,
                tokenizer=tokenizer,
                code=code,
                device=device,
                max_src_len=256,
                max_gen_len=64,
            )

            print("\nGenerated Summary:")
            print(summary)
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
