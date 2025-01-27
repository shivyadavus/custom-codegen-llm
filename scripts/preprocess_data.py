"""
preprocess_data.py - Script to clean and prepare code/text files for LLM training.
"""

import os
import re

RAW_DATA_DIR = "../data_code"
PROCESSED_DATA_DIR = "../data_code"  # Could also set to another directory if desired

def basic_clean(text: str) -> str:
    """
    Perform minimal cleaning on the text, e.g., removing extra whitespace.
    Extend this function as needed for your domain (e.g., removing secrets or tokens).
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def process_files():
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                # Apply cleaning
                cleaned_text = basic_clean(text)
                
                # Example: simply overwrite with cleaned content
                # or write to a separate directory if you prefer
                processed_path = os.path.join(PROCESSED_DATA_DIR, filename)
                with open(processed_path, "w", encoding="utf-8") as outf:
                    outf.write(cleaned_text)
                
                print(f"Processed {file_path} -> {processed_path}")

if __name__ == "__main__":
    process_files()
