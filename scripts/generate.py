import sys
import os
import torch

sys.path.append(os.path.abspath("."))

from src.inference import load_inference_model, summarize_code


def generate_summary():
    model_path = "/content/drive/MyDrive/ml-python-code-summarization/models/best.pt"
    tokenizer_path = "data/tokenizer/tokenizer.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Model: {model_path}")

    model, tokenizer = load_inference_model(model_path, tokenizer_path, device)

    print("\n✅ Model loaded. Paste code. Finish with an empty line. Ctrl+C to exit.")
    print("-" * 60)

    while True:
        try:
            print(">> Enter Code (end with empty line):")
            lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            code = "\n".join(lines).strip()
            if not code:
                continue

            summary = summarize_code(model, tokenizer, code, device, max_src_len=256, max_gen_len=64)
            print("\nGenerated Summary:")
            print(summary)
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    generate_summary()
