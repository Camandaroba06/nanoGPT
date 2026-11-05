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

    full_tokens = []

    for text in lines:
        full_tokens.extend(enc.encode_ordinary(text))
        full_tokens.append(enc.eot_token)

    chunk_tokens = []

    for chunk in iter_token_chunks(lines):
        chunk_tokens.extend(chunk)

    if full_tokens == chunk_tokens:
        print("Tokenização equivalente no teste.")
        print(f"Documentos processados: {len(lines):,}")
        print(f"Tokens: {len(chunk_tokens):,}")
        print(f"Tokens EOS: {chunk_tokens.count(enc.eot_token):,}")
    else:
        print("ATENÇÃO: diferenças na tokenização.")

        for i, (a, b) in enumerate(zip(full_tokens, chunk_tokens)):
            if a != b:
                print(f"Primeira divergência na posição {i}:")
                print(f"Token de referência: {a}")
                print(f"Token em blocos: {b}")
                break

        print(f"Tokens de referência: {len(full_tokens):,}")
        print(f"Tokens em blocos: {len(chunk_tokens):,}")
        
def iter_token_chunks(texts, chunk_size=CHUNK_SIZE):
    token_buffer = []

    for text in texts:
        document_tokens = enc.encode_ordinary(text)
        document_tokens.append(enc.eot_token)

        # Finaliza o bloco antes de adicionar um documento que o ultrapassaria.
        if token_buffer and len(token_buffer) + len(document_tokens) > chunk_size:
            yield token_buffer
            token_buffer = []

        token_buffer.extend(document_tokens)

        # O bloco é finalizado somente após um documento completo.
        if len(token_buffer) >= chunk_size:
            yield token_buffer
            token_buffer = []

    if token_buffer:
        yield token_buffer

def prepare_split(split, max_tokens):
    output_path = os.path.join(OUTPUT_DIR, f"{split}.bin")
    total_tokens = 0

    texts = (
        example["text"]
        for example in dataset[split].shuffle(
            seed=42,
            buffer_size=10_000
        )
    )

    with open(output_path, "wb") as output_file:
        for token_chunk in iter_token_chunks(texts):
            remaining = max_tokens - total_tokens

            if remaining <= 0:
                break

            if len(token_chunk) > remaining:
                token_chunk = token_chunk[:remaining]

                if token_chunk:
                    token_chunk[-1] = enc.eot_token

            token_array = np.asarray(token_chunk, dtype=np.uint16)
            token_array.tofile(output_file)

            total_tokens += len(token_array)

            if total_tokens >= max_tokens:
                print(f"Limite de {max_tokens:,} tokens atingido para '{split}'.")
                break

    print(f"{split}: {total_tokens:,} tokens")
    return total_tokens

if __name__ == "__main__":
    print("Executando teste de sanidade da tokenização (1000 documentos)...")
    validate_chunking("train", num_lines=1000)

    splits = {"train": MAX_TOKENS_TRAIN, "validation": MAX_TOKENS_VAL}

    for split, max_tokens in splits.items():
        print(f"\nPreparando {split}...")
        prepare_split(split, max_tokens)

    print("\nC4 preparation completed!")
