import os
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
):
    """
    Load tokenizer + model checkpoint (best.pt or last.pt).

    The model hyperparameters MUST match training.
    """
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
        emb_dim=emb_dim,
        enc_hidden=enc_hidden,
        dec_hidden=dec_hidden,
        num_layers=num_layers,
        dropout=0.0,  # eval only; dropout disabled anyway
        pad_id=pad_id,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, tokenizer


def _encode_source(
    tokenizer: Tokenizer,
    code: str,
    max_src_len: int,
    pad_id: int,
    device: str,
):
    """
    Encode code -> (src_ids, src_mask) exactly like training:
      - truncate to max_src_len
      - pad to max_src_len
      - mask = 1 for real tokens, 0 for pad
    """
    enc = tokenizer.encode(code)
    ids = enc.ids[:max_src_len]
    length = len(ids)

    if length < max_src_len:
        ids = ids + [pad_id] * (max_src_len - length)

    src_ids = torch.tensor([ids], dtype=torch.long, device=device)
    src_mask = torch.tensor([[1] * length + [0] * (max_src_len - length)], dtype=torch.long, device=device)
    return src_ids, src_mask


def summarize_code(
    model,
    tokenizer: Tokenizer,
    code: str,
    device: str,
    max_src_len: int = 256,
    max_len: int = 64,
) -> str:
    """
    Greedy generation using model.generate(...)

    max_src_len must match training (256 in your fast setup).
    max_len should match your target length (64 is consistent).
    """
    code = (code or "").strip()
    if not code:
        return ""

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")

    if bos_id is None or eos_id is None or pad_id is None:
        raise ValueError("Tokenizer must include <bos>, <eos>, <pad>")

    src_ids, src_mask = _encode_source(tokenizer, code, max_src_len, pad_id, device)

    with torch.no_grad():
        generated = model.generate(
            src_ids=src_ids,
            src_mask=src_mask,
            max_len=max_len,
            bos_id=bos_id,
            eos_id=eos_id,
        )

    # generated could be: list[int], tensor[T], tensor[1,T]
    if isinstance(generated, torch.Tensor):
        if generated.dim() == 2:
            generated_ids = generated[0].tolist()
        else:
            generated_ids = generated.tolist()
    else:
        generated_ids = list(generated)

    # Decode
    summary = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return summary.strip()
