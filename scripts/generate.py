import sys
import os
import torch
from tokenizers import Tokenizer

# Ensure we can import from src
sys.path.append(os.path.abspath("."))

from src.model import Seq2SeqLSTMAttn

def generate_summary():
    # settings
    model_path = "models/best.pt"
    tokenizer_path = "data/tokenizer/tokenizer.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Train the model first!")
        return

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_id = tokenizer.token_to_id("<pad>")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    vocab_size = tokenizer.get_vocab_size()

    print(f"Loading model from {model_path}...")
    # Matches train.py params
    model = Seq2SeqLSTMAttn(
        vocab_size=vocab_size, 
        emb_dim=256, 
        enc_hidden=256, 
        dec_hidden=512, 
        num_layers=1, 
        dropout=0.0, # dropout irrelevant for eval
        pad_id=pad_id
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\nModel loaded. Enter Python code snippet to summarize (Ctrl+C to exit):")
    print("-" * 50)
    
    while True:
        try:
            # Multi-line input
            print(">> Enter Code (end with empty line):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            
            code = "\n".join(lines)
            if not code.strip():
                continue
                
            # Preprocess
            encoded = tokenizer.encode(code)
            src_ids = torch.tensor([encoded.ids], dtype=torch.long).to(device)
            # Mask (all 1s since we aren't batching/padding here for single input)
            src_mask = torch.ones_like(src_ids).to(device)
            
            # Generate
            with torch.no_grad():
                generated_ids = model.generate(src_ids, src_mask, max_len=50, bos_id=bos_id, eos_id=eos_id)
            
            # Decode
            summary = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"\nGenerared Summary: {summary}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    generate_summary()
