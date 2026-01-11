import sys
import os
import torch

# Ensure we can import from src
sys.path.append(os.path.abspath("."))

from src.inference import load_inference_model, summarize_code

def generate_summary():
    # settings
    model_path = "models/best.pt"
    tokenizer_path = "data/tokenizer/tokenizer.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path} and tokenizer from {tokenizer_path}...")
    try:
        model, tokenizer = load_inference_model(model_path, tokenizer_path, device)
    except FileNotFoundError as e:
        print(e)
        print("Please train the model first!")
        return

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
            
            # Use the inference function
            summary = summarize_code(model, tokenizer, code, device)
            
            print(f"\nGenerated Summary: {summary}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    generate_summary()
