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
