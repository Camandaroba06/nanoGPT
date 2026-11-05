
import os
import numpy as np
import tiktoken
from datasets import load_dataset

DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_NAME = "gpt2"
CHUNK_SIZE = 1_000_000

print("Loading WikiText-103...")

dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

print("Dataset loaded successfully.")

enc = tiktoken.get_encoding(TOKENIZER_NAME)

print(f"Tokenizer: {TOKENIZER_NAME}")
print(f"Vocabulary size: {enc.n_vocab:,}")

def iter_text_chunks(split, chunk_size=CHUNK_SIZE):
    buffer = []
    current_size = 0
    first_line = True

    for example in dataset[split]:

        text = example["text"]

        if not first_line:
            text = "\n" + text

        first_line = False

        if buffer and current_size + len(text) > chunk_size:
            yield "".join(buffer)

            buffer = []
            current_size = 0

        buffer.append(text)
        current_size += len(text)

    if buffer:
        yield "".join(buffer)

def prepare_split(split):
    output_path = os.path.join(OUTPUT_DIR, f"{split}.bin")
    total_tokens = 0

    print(f"\nPreparing {split} split...")

    with open(output_path, "wb") as output_file:
        for chunk in iter_text_chunks(split):
            token_ids = enc.encode_ordinary(chunk)
            token_array = np.asarray(token_ids, dtype=np.uint16)

            token_array.tofile(output_file)
            total_tokens += len(token_array)

    print(f"{split}: {total_tokens:,} tokens")
    print(f"Saved to: {output_path}")

    return total_tokens

if __name__ == "__main__":
    splits = ["train", "validation", "test"]

    for split in splits:
        prepare_split(split)

    print("\nWikiText-103 preparation completed!")
