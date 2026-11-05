# Training Compute-Optimal Large Language Models (Hoffmann et al., 2022, DeepMind) - parametros modelo × 10 = número tokens necessários 

import os
import numpy as np
import tiktoken
from datasets import load_dataset

DATASET_NAME = "allenai/c4"
DATASET_CONFIG = "en"

MAX_TOKENS_TRAIN = 3_000_000_000  
MAX_TOKENS_VAL = 100_000_000 

base_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(base_dir, 'data', 'c4')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOKENIZER_NAME = "gpt2"
CHUNK_SIZE = 1_000_000

print("Loading C4 in streaming mode...")
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, streaming=True)
print("Dataset loaded successfully.")

enc = tiktoken.get_encoding(TOKENIZER_NAME)

print(f"Tokenizer: {TOKENIZER_NAME}")
print(f"Vocabulary size: {enc.n_vocab:,}")

def iter_text_chunks_from_lines(lines, chunk_size=CHUNK_SIZE):
    buffer = []
    current_size = 0

    for line in lines:
        buffer.append(line)
        current_size += len(line) + 1

        if current_size >= chunk_size:
            yield "\n".join(buffer) + "\n"
            buffer = []
            current_size = 0

    if buffer:
        yield "\n".join(buffer)

def validate_chunking(split, num_lines=1000):
    lines = [
        example["text"]
        for example in dataset[split].take(num_lines)
    ]

    # Tokenização do texto completo 
    full_text = "\n".join(lines)
    full_tokens = enc.encode_ordinary(full_text)

    chunk_tokens = []

    for chunk in iter_text_chunks_from_lines(lines):
        chunk_tokens.extend(enc.encode_ordinary(chunk))

    if full_tokens == chunk_tokens:
        print("Tokenização equivalente no teste.")
    else:
        print("ATENÇÃO: diferenças na tokenização.")

        # primeira posição divergente
        for i, (a, b) in enumerate(zip(full_tokens, chunk_tokens)):
            if a != b:
                print(f"Primeira divergência na posição {i}:")
                print(f"Token integral: {a}")
                print(f"Token em chunks: {b}")
                break

        print(f"Tokens no texto integral: {len(full_tokens)}")
        print(f"Tokens em chunks: {len(chunk_tokens)}")
        
def iter_text_chunks(split, chunk_size=CHUNK_SIZE):
    buffer = []
    current_size = 0

    for example in dataset[split]:
        line = example["text"]

        buffer.append(line)
        current_size += len(line) + 1  

        if current_size >= chunk_size:
            yield "\n".join(buffer) + "\n"
            buffer = []
            current_size = 0

    if buffer:
        yield "\n".join(buffer)

def prepare_split(split, max_tokens):
    output_path = os.path.join(OUTPUT_DIR, f"{split}.bin")
    total_tokens = 0

    with open(output_path, "wb") as output_file:
        for chunk in iter_text_chunks(split):
            token_ids = enc.encode_ordinary(chunk)
            token_array = np.asarray(token_ids, dtype=np.uint16)

            token_array.tofile(output_file)
            total_tokens += len(token_array)

            # 5. Condição de parada adicionada para respeitar o limite de tokens
            if total_tokens >= max_tokens:
                print(f"Limite de {max_tokens:,} tokens atingido para '{split}'.")
                break

    print(f"{split}: {total_tokens:,} tokens")
    return total_tokens

if __name__ == "__main__":
    print("Executando teste de sanidade da tokenização (1000 linhas)...")
    validate_chunking("train", num_lines=1000)

    # splits = {"train": MAX_TOKENS_TRAIN, "validation": MAX_TOKENS_VAL}

    # for split, max_tokens in splits.items():
    #    print(f"\nPreparando {split}...")
    #    prepare_split(split, max_tokens)

    # print("\nC4 preparation completed!")
