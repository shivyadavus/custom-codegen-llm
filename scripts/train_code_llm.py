"""
Fine-tune distilgpt2 on local text and compute perplexity manually.
"""

import os
import math
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)


def compute_perplexity(model, tokenizer, raw_dataset, max_length=128, subset_size=100):
    """
    Manually compute perplexity:
      1) Select up to subset_size lines from `raw_dataset`.
      2) For each line, tokenize and run a forward pass with labels=input_ids.
      3) Accumulate loss * number_of_tokens, track total tokens.
      4) average_loss = total_loss / total_tokens
      5) perplexity = exp(average_loss)
    """
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optionally take a small subset for speed
    raw_subset = raw_dataset.select(range(min(subset_size, len(raw_dataset))))

    total_loss = 0.0
    total_tokens = 0

    for row in raw_subset:
        text = row["text"]
        # Tokenize
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(device)
        # Some tokenizers also produce attention_mask
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Forward pass with labels=input_ids for causal LM
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
        loss = outputs.loss  # average loss per token in this batch
        num_tokens = input_ids.size(1)

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    if total_tokens == 0:
        return float("inf")

    average_loss = total_loss / total_tokens
    ppl = math.exp(average_loss)
    return ppl


def main():
    # ---------------------------
    # A. Load Pretrained distilgpt2
    # ---------------------------
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # GPT-2 doesn't define a pad_token by default => set to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # ---------------------------
    # B. Load Local Text Data
    # ---------------------------
    # Put .txt files in ../data_code/ if running from scripts/
    data_files = {"train": "../data_code/*.txt"}
    raw_dataset = load_dataset("text", data_files=data_files)
    train_dataset_raw = raw_dataset["train"]

    # ---------------------------
    # C. Tokenize
    # ---------------------------
    # We'll keep a column "text" in raw_dataset for the perplexity function.
    max_seq_length = 128

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length
        )

    # Note: We won't remove_columns=["text"], so the raw text is still available
    tokenized_dataset = train_dataset_raw.map(
        tokenize_fn,
        batched=True
    )

    # ---------------------------
    # D. Data Collator
    # ---------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # ---------------------------
    # E. Training Arguments
    # ---------------------------
    training_args = TrainingArguments(
        output_dir="../checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=2,             # Increase if you have more data
        per_device_train_batch_size=2,  # Adjust to your CPU/GPU
        save_steps=100,
        logging_steps=50,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available()
    )

    # ---------------------------
    # F. Trainer
    # ---------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # ---------------------------
    # G. Train
    # ---------------------------
    trainer.train()

    # ---------------------------
    # H. Manual Perplexity
    # ---------------------------
    # This approach is guaranteed not to cause "ValueError" from caching.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ppl_value = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        raw_dataset=train_dataset_raw,  # use same training set or create a val set
        max_length=max_seq_length,
        subset_size=100
    )

    print("\n=== Manual Perplexity ===")
    print(f"Perplexity: {ppl_value:.2f}")

    # ---------------------------
    # I. Sample Generation
    # ---------------------------
    model.eval()
    prompt_text = "def add_numbers(a, b):\n    # This function adds two numbers\n"
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            no_repeat_ngram_size=2
        )

    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("\n=== Sample Completion ===")
    print("Prompt:\n", prompt_text)
    print("Completion:\n", output_text)


if __name__ == "__main__":
    main()
