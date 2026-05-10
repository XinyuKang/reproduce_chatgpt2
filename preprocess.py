# scripts/prepare_openwebtext2.py

import os
import json
import glob
import zstandard as zstd
import numpy as np
import tiktoken
from tqdm import tqdm


RAW_DIR = "data/raw/openwebtext2/openwebtext2.jsonl.zst"
OUT_DIR = "data/processed"
TRAIN_PATH = os.path.join(OUT_DIR, "train.bin")
VAL_PATH = os.path.join(OUT_DIR, "val.bin")

VAL_RATIO = 0.001  # small validation split
MAX_DOCS = 100_000  # start small first; later set to None


def read_zst_jsonl(path):
    dctx = zstd.ZstdDecompressor()

    with open(path, "rb") as f:
        with dctx.stream_reader(f) as reader:
            buffer = b""

            while True:
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break

                buffer += chunk
                lines = buffer.split(b"\n")
                buffer = lines[-1]

                for line in lines[:-1]:
                    if line.strip():
                        yield json.loads(line)

            if buffer.strip():
                yield json.loads(buffer)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.jsonl.zst")))

    print(f"Found {len(files)} shard files")

    all_tokens = []
    doc_count = 0

    for path in files:
        print(f"Processing {path}")

        for obj in tqdm(read_zst_jsonl(path)):
            text = obj.get("text", "")

            if not text:
                continue

            tokens = enc.encode(text)
            tokens.append(eot)

            all_tokens.extend(tokens)

            doc_count += 1

            if MAX_DOCS is not None and doc_count >= MAX_DOCS:
                break

        if MAX_DOCS is not None and doc_count >= MAX_DOCS:
            break

    all_tokens = np.array(all_tokens, dtype=np.uint16)

    split_idx = int(len(all_tokens) * (1 - VAL_RATIO))

    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    train_tokens.tofile(TRAIN_PATH)
    val_tokens.tofile(VAL_PATH)

    print(f"Documents processed: {doc_count:,}")
    print(f"Total tokens:        {len(all_tokens):,}")
    print(f"Train tokens:        {len(train_tokens):,}")
    print(f"Val tokens:          {len(val_tokens):,}")
    print(f"Saved to {TRAIN_PATH}")
    print(f"Saved to {VAL_PATH}")


if __name__ == "__main__":
    main()