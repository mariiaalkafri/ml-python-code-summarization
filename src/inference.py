import os
from typing import Tuple, List

import torch
from tokenizers import Tokenizer

from src.model import Seq2SeqLSTMAttn


def load_inference_model(
    model_path: str,import os
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

    # MUST match training hyperparams
    model = Seq2SeqLSTMAttn(
        vocab_size=vocab_size,
        emb_dim=256,
        enc_hidden=256,
        dec_hidden=512,
        num_layers=1,
        dropout=0.0,  # eval
        pad_id=pad_id,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model, tokenizer


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

    enc = tokenizer.encode(code)
    ids = enc.ids[:max_src_len]

    # pad + mask
    length = len(ids)
    if length < max_src_len:
        ids = ids + [pad_id] * (max_src_len - length)
    src_ids = torch.tensor([ids], dtype=torch.long, device=device)  # [1,S]
    src_mask = (src_ids != pad_id).long()  # [1,S]

    with torch.no_grad():
        sequences = model.generate(
            src_ids=src_ids,
            src_mask=src_mask,
            max_len=max_gen_len,
            bos_id=bos_id,
            eos_id=eos_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
        )

    # sequences[0] includes BOS and generated tokens
    gen_ids = sequences[0]

    # remove BOS, cut at EOS
    if len(gen_ids) > 0 and gen_ids[0] == bos_id:
        gen_ids = gen_ids[1:]
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]

    summary = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return summary

    tokenizer_path: str,
    device: str,
    emb_dim: int = 256,
    enc_hidden: int = 256,
    dec_hidden: int = 512,
    num_layers: int = 1,
) -> Tuple[Seq2SeqLSTMAttn, Tokenizer]:
    """
    Load trained model + tokenizer for inference.
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
        raise ValueError("Tokenizer must contain <pad>, <bos>, <eos>")

    model = Seq2SeqLSTMAttn(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        enc_hidden=enc_hidden,
        dec_hidden=dec_hidden,
        num_layers=num_layers,
        dropout=0.0,
        pad_id=pad_id,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)

    model.eval()
    return model, tokenizer


def _strip_special_ids(ids: List[int], pad_id: int, bos_id: int, eos_id: int) -> List[int]:
    """Remove PAD/BOS and stop at EOS."""
    cleaned = []
    for t in ids:
        if t == eos_id:
            break
        if t in (pad_id, bos_id):
            continue
        cleaned.append(t)
    return cleaned


def _collapse_repetitions(tokens: List[int], max_repeat: int = 3) -> List[int]:
    """Avoid ugly loops like 'b b b b b'."""
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
    Generate summary for a single code snippet.
    """

    code = code.strip()
    if not code:
        return ""

    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    enc = tokenizer.encode(code)
    src = enc.ids[:max_src_len]

    if len(src) < max_src_len:
        src += [pad_id] * (max_src_len - len(src))

    src_ids = torch.tensor([src], dtype=torch.long, device=device)
    src_mask = (src_ids != pad_id).long()

    generated = model.generate(
        src_ids=src_ids,
        src_mask=src_mask,
        max_len=max_gen_len,
        bos_id=bos_id,
        eos_id=eos_id,
    )

    if isinstance(generated, torch.Tensor):
        generated = generated.squeeze(0).tolist()

    generated = _strip_special_ids(generated, pad_id, bos_id, eos_id)
    generated = _collapse_repetitions(generated)

    if not generated:
        return ""

    return tokenizer.decode(generated).strip()
