"""
train_code_llm.py - Main script to train a GPT-like model on your code.
"""

import math
import torch
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

def main():
    # 1. Load or create a tokenizer specialized for code
    #    (If you have a custom tokenizer, place its JSON in ../tokenizer)
    #    Otherwise, you could load a pretrained tokenizer like CodeGPT or CodeGen.
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="../tokenizer/my_code_tokenizer.json")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # 2. Load dataset from local code snippets
    dataset = load_dataset("text", data_files={"train": "../data_code/*.txt"})
    train_dataset = dataset["train"]

    # 3. Tokenize
    max_seq_length = 1024
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length
            # 'return_overflowing_tokens' can also be True if you want chunking
        )

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 4. Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5. Configure a GPT-2 style model
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12
    )
    model = GPT2LMHeadModel(config)

    # 6. Training arguments
    training_args = TrainingArguments(
        output_dir="../checkpoints",
        num_train_epochs=2,  # Increase or adjust for your needs
        per_device_train_batch_size=2,
        save_steps=500,
        logging_steps=100,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available()
    )

    # 7. Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    # 8. Train
    trainer.train()

    # 9. Quick generation test
    model.eval()
    prompt = "def parse_requirements(file_path):"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        model.cuda()
        input_ids = input_ids.cuda()

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_length=100,
            num_beams=5,
            temperature=1.0,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("\n--- Model Output ---")
    print(output_text)

if __name__ == "__main__":
    main()
