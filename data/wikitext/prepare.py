# acho que é melhor usar o wikitest somente para debug por conta do risco de overfitting
import os
import numpy as np
import tiktoken
from datasets import load_dataset

DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"

# usei o wikitest 103 raw porque as palavras originais são preservadas (no wikitest 103 processado eles
# substituem palavras raras por <unk>

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_NAME = "gpt2"
CHUNK_SIZE = 1_000_000

print("Loading WikiText-103...")

dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

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
        for example in dataset[split].select(range(num_lines))
    ]

    # Tokenização do texto completo 
    full_text = "\n".join(lines)
    full_tokens = enc.encode_ordinary(full_text)

    # Tokenização em chunks
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

def prepare_split(split):
    output_path = os.path.join(OUTPUT_DIR, f"{split}.bin")
    total_tokens = 0

    with open(output_path, "wb") as output_file:
        for chunk in iter_text_chunks(split):
            token_ids = enc.encode_ordinary(chunk)
            token_array = np.asarray(token_ids, dtype=np.uint16)

            token_array.tofile(output_file)
            total_tokens += len(token_array)

    print(f"{split}: {total_tokens:,} tokens")
    return total_tokens

if __name__ == "__main__":
    print("Executando teste de sanidade da tokenização...")
    validate_chunking("train", num_lines=1000) 
    
    # splits = ["train", "validation", "test"]
    # for split in splits:
    #    prepare_split(split)

    # print("\nWikiText-103 preparation completed!")
