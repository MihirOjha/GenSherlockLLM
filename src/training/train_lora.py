"""
train_lora.py
-------------
Fine-tunes a small GPT-style model using LoRA on Sherlock Holmes chunks.

Usage:
    python src/training/train_lora.py
"""

import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

# -------------------- Paths --------------------
CLEAN_DIR = Path("data/cleaned")
OUTPUT_DIR = Path("experiments/sherlock_lora")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Hyperparameters --------------------
BASE_MODEL = "gpt2"        # small model for first experiment
MAX_LENGTH = 200           # matches chunk size
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-4
ADAPTER_R = 8              # LoRA rank

# -------------------- Load Dataset --------------------
def load_dataset():
    """Load all JSON chunk files into a Hugging Face Dataset"""
    data = []
    for file in CLEAN_DIR.glob("*.json"):
        with file.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
            data.extend([{"text": c["text"]} for c in chunks])
    return Dataset.from_list(data)


# -------------------- Tokenizer & Preprocessing --------------------
def tokenize_fn(batch, tokenizer):
    """Tokenize text for causal language modeling"""
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# -------------------- Main Training Pipeline --------------------
def main():
    print("📥 Loading dataset...")
    dataset = load_dataset()
    print(f"✅ Loaded {len(dataset)} chunks.")

    print("📥 Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # GPT2 does not have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    print("📝 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_fn(x, tokenizer), 
        batched=True,
        remove_columns=dataset.column_names  # Remove original text columns to avoid conflicts
    )

    # -------------------- LoRA Setup --------------------
    print("⚡ Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=ADAPTER_R,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],  # GPT2 attention and output projections
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # -------------------- Training --------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        fp16=True,               # if GPU supports
        eval_strategy="no",
        remove_unused_columns=True,  # Set to True to avoid issues
        dataloader_pin_memory=False,  # Disable if no GPU
    )

    # Use a data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("🚀 Starting fine-tuning...")
    trainer.train()

    print("💾 Saving LoRA adapter & model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()