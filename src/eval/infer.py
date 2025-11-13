"""
infer.py
--------
Loads the fine-tuned LoRA Sherlock model and generates text given a prompt.

Usage:
    python src/eval/infer.py --prompt "On human nature and perseverance"
"""

import torch
from pathlib import Path
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- Paths --------------------
MODEL_DIR = Path("experiments/sherlock_lora")

# -------------------- Inference Function --------------------
def generate_text(prompt: str, max_new_tokens: int = 80, temperature: float = 0.9, top_p: float = 0.9):
    """Generate Sherlock-style text from the fine-tuned model"""
    print("🔍 Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Encode input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            temperature=0.7,         # lower = more deterministic
            top_p=0.92,              # nucleus sampling
            top_k=50,                # avoid extreme randomness
            do_sample=True,
            repetition_penalty=1.3,  # discourages loops
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode output
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n🧠 Sherlock Wisdom:\n")
    print(generated)
    print("\n-------------------------------------------")

# -------------------- CLI Entry Point --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to guide text generation")
    args = parser.parse_args()

    generate_text(args.prompt)
