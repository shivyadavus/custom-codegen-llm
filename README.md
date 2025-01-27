
# My Custom LLM Codegen

A GPT-style model trained on my GitHub repositories for specialized code completion and Q&A. 

## Environment Setup
1. Install Python 3.8+
2. Create and activate a virtual environment:
   ```bash
   python -m venv llm_env
   source llm_env/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Repository Structure
- **data_code/**: Contains .txt files representing code or documentation.
- **scripts/**:
  - **train_code_llm.py**: Main training script for the model.
  - **preprocess_data.py**: Optional script for data cleaning/splitting.
- **tokenizer/**: Holds tokenizer files (e.g., `my_code_tokenizer.json`).
- **checkpoints/**: Model checkpoints and logs (ignored by default in .gitignore).

## Usage
1. Preprocess your data (if applicable):
   ```bash
   python scripts/preprocess_data.py
   ```
2. Run the training:
   ```bash
   python scripts/train_code_llm.py
   ```

## License
MIT

## References
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [Article on building an LLM](link-to-your-article)
