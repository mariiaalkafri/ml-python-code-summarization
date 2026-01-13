import os
import torch
from tokenizers import Tokenizer
from src.transformer_model import TransformerSeq2Seq


def load_inference_model_transformer(model_path: str, tokenizer_path: str, device: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise ValueError("Tokenizer missing <pad> token")

    # IMPORTANT: must match training config exactly
    model = TransformerSeq2Seq(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
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


def summarize_code_transformer(
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

    # Encode + pad source (same as LSTM)
    enc = tokenizer.encode(code)
    ids = enc.ids[:max_src_len]
    if len(ids) < max_src_len:
        ids = ids + [pad_id] * (max_src_len - len(ids))

    src_ids = torch.tensor([ids], dtype=torch.long, device=device)
    src_mask = (src_ids != pad_id).long()

    with torch.no_grad():
        sequences = model.generate(
            src_ids=src_ids,
            src_mask=src_mask,
            max_len=max_gen_len,
            bos_id=bos_id,
            eos_id=eos_id,
        )

    # model.generate returns batch of token id lists
    gen_ids = sequences[0]

    # remove BOS
    if gen_ids and gen_ids[0] == bos_id:
        gen_ids = gen_ids[1:]

    # cut at EOS
    if eos_id in gen_ids:
        gen_ids = gen_ids[: gen_ids.index(eos_id)]

    # decode
    summary = tokenizer.decode(gen_ids).strip()
    return summary
