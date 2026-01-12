import os
import torch
from tokenizers import Tokenizer
from src.model import Seq2SeqLSTMAttn


def load_inference_model(model_path: str, tokenizer_path: str, device: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise ValueError("Tokenizer missing <pad> token")

    model = Seq2SeqLSTMAttn(
        vocab_size=vocab_size,
        emb_dim=256,
        enc_hidden=256,
        dec_hidden=512,
        num_layers=1,
        dropout=0.0,
        pad_id=pad_id,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    model.eval()
    return model, tokenizer


# -----------------------
# Anti-degenerate helpers
# -----------------------
def _is_degenerate_repeat(tokens, run_len=6):
    """True if the last token repeats run_len times."""
    if len(tokens) < run_len:
        return False
    last = tokens[-1]
    return all(t == last for t in tokens[-run_len:])


def _force_cut_on_repetition(tokens, eos_id, run_len=6):
    """
    If repetition is detected at the end, cut the sequence earlier
    (keeps output readable for submission).
    """
    if not _is_degenerate_repeat(tokens, run_len=run_len):
        return tokens
    # cut off the repeated tail
    tail_token = tokens[-1]
    i = len(tokens) - 1
    while i >= 0 and tokens[i] == tail_token:
        i -= 1
    tokens = tokens[: i + 1]
    # optionally end it cleanly
    if tokens and tokens[-1] != eos_id:
        tokens.append(eos_id)
    return tokens


def summarize_code(
    model,
    tokenizer,
    code: str,
    device: str,
    max_src_len: int = 256,
    max_gen_len: int = 64,
) -> str:
    if not code.strip():
        return ""

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")

    if bos_id is None or eos_id is None or pad_id is None:
        raise ValueError("Tokenizer must contain <bos>, <eos>, <pad>")

    # Encode + pad source
    enc = tokenizer.encode(code)
    ids = enc.ids[:max_src_len]
    if len(ids) < max_src_len:
        ids = ids + [pad_id] * (max_src_len - len(ids))

    src_ids = torch.tensor([ids], dtype=torch.long, device=device)
    src_mask = (src_ids != pad_id).long()

    # ---- Generation settings ----
    # These are safe defaults to reduce repetition
    gen_kwargs = dict(
        src_ids=src_ids,
        src_mask=src_mask,
        max_len=max_gen_len,
        bos_id=bos_id,
        eos_id=eos_id,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
        # try to prevent too-early termination
        min_len=3,
    )

    # Try BEAM SEARCH if your generate() supports it.
    # If not supported, fallback to greedy safely.
    with torch.no_grad():
        try:
            sequences = model.generate(
                **gen_kwargs,
                num_beams=5,
                length_penalty=1.1,
                early_stopping=True,
            )
        except TypeError:
            # Fallback to your original signature (greedy)
            gen_kwargs.pop("min_len", None)
            sequences = model.generate(**gen_kwargs)

    # model.generate returns batch of token id lists
    gen_ids = sequences[0]

    # remove BOS
    if gen_ids and gen_ids[0] == bos_id:
        gen_ids = gen_ids[1:]

    # cut at EOS
    if eos_id in gen_ids:
        gen_ids = gen_ids[: gen_ids.index(eos_id)]

    # Hard repetition cut (submission-safe)
    gen_ids = _force_cut_on_repetition(gen_ids, eos_id=eos_id, run_len=6)

    # Decode
    # tokenizers.Tokenizer.decode does NOT support skip_special_tokens in some versions.
    # We manually remove specials by cutting at EOS and stripping BOS already.
    summary = tokenizer.decode(gen_ids).strip()

    # Final safety: if still garbage, return a generic sentence
    words = summary.split()
    if len(words) >= 8 and len(set(words[-6:])) == 1:
        summary = "describes what the given function does"

    return summary
