import torch
import os
from tokenizers import Tokenizer
from src.model import Seq2SeqLSTMAttn

def load_inference_model(model_path: str, tokenizer_path: str, device: str):
    """
    Loads the trained model and tokenizer.
    
    Args:
        model_path: Path to the .pt checkpoint file.
        tokenizer_path: Path to the tokenizer.json file.
        device: 'cuda' or 'cpu'.
        
    Returns:
        model: The loaded Seq2SeqLSTMAttn model (eval mode).
        tokenizer: The loaded Tokenizer.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
    # Load Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("<pad>")
    
    # Initialize Model - matching training hyperparameters
    model = Seq2SeqLSTMAttn(
        vocab_size=vocab_size, 
        emb_dim=256, 
        enc_hidden=256, 
        dec_hidden=512, 
        num_layers=1, 
        dropout=0.0, # dropout irrelevant for eval
        pad_id=pad_id
    ).to(device)
    
    # Load Weights
    checkpoint = torch.load(model_path, map_location=device)
    # Handle cases where checkpoint might be nested
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    return model, tokenizer

def summarize_code(model, tokenizer, code: str, device: str, max_len: int = 50) -> str:
    """
    Generates a summary for a given code snippet.
    
    Args:
        model: The loaded Seq2SeqLSTMAttn model.
        tokenizer: The loaded Tokenizer.
        code: The Python code string to summarize.
        device: 'cuda' or 'cpu'.
        max_len: Maximum length of the generated summary.
        
    Returns:
        summary: The generated summary string.
    """
    if not code.strip():
        return ""
        
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    
    # Preprocess
    encoded = tokenizer.encode(code)
    src_ids = torch.tensor([encoded.ids], dtype=torch.long).to(device)
    
    # Simple masking (all 1s as we process single input without padding)
    src_mask = torch.ones_like(src_ids).to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(src_ids, src_mask, max_len=max_len, bos_id=bos_id, eos_id=eos_id)
    
    # Decode
    summary = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return summary
