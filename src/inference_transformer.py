import torch
import os
from tokenizers import Tokenizer
from src.transformer_model import TransformerSeq2Seq

def load_transformer_model(model_path: str, tokenizer_path: str, device: str):
    """
    Loads the trained Transformer model and tokenizer.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
    # Load Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<pad>")
    
    # Initialize Model
    model = TransformerSeq2Seq(
        vocab_size=vocab_size, 
        d_model=256, 
        nhead=8, 
        num_encoder_layers=4, 
        num_decoder_layers=4, 
        dim_feedforward=1024,
        dropout=0.0, 
        pad_id=pad_id
    ).to(device)
    
    # Load Weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    return model, tokenizer

def summarize_code_transformer(model, tokenizer, code: str, device: str, max_len: int = 50) -> str:
    """
    Generates a summary for a given code using Transformer model.
    """
    if not code.strip():
        return ""
        
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    
    encoded = tokenizer.encode(code)
    src_ids = torch.tensor([encoded.ids], dtype=torch.long).to(device)
    
    # src_mask (1 for valid)
    src_mask = torch.ones_like(src_ids).to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(src_ids, src_mask, max_len=max_len, bos_id=bos_id, eos_id=eos_id)
    
    summary = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return summary
