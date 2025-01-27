from tokenizers import Tokenizer, trainers, models
import os
import glob

def train_my_tokenizer(data_dir="data_code", vocab_size=32000, output_file="tokenizer/my_code_tokenizer.json"):
    # 1. Initialize a BPE-based tokenizer with an UNK token
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # 2. Set up a BPE Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    # 3. Gather all .txt files to learn from
    text_files = glob.glob(os.path.join(data_dir, "*.txt"))

    # 4. Train the tokenizer on these files
    tokenizer.train(files=text_files, trainer=trainer)

    # 5. Save the tokenizer to disk
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tokenizer.save(output_file)

if __name__ == "__main__":
    train_my_tokenizer()
