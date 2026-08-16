# Streams the nob_Latn split of FineWeb 2, filters it, tokenizes it with
# the existing tokenizer and writes the training shards, val set and word
# list into public/. Run: uv run --with "tokenizers,numpy,pyarrow,fsspec,aiohttp,requests" python scripts/build_corpus.py
import collections
import os
import json
import re

import fsspec
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

PUB = os.path.join(os.path.dirname(__file__), "..", "public")
TARGET_CHARS = 400_000_000
VAL_CHARS = 6_000_000
# small shards so the page starts training after one quick download, and the
# CDN caches every gen-named file immutably
SHARD_TOKENS = 4_000_000
GEN = 2

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

# rebuild the EXACT tokenizer from the exported vocab/merges so token ids match
byte_encoder = bytes_to_unicode()
tk = json.load(open(f"{PUB}/tok16k-g{GEN}.json"))
id_to_str = ["".join(byte_encoder[b] for b in bts) for bts in tk["vocab"]]
vocab = {s: i for i, s in enumerate(id_to_str)}
merges = [(id_to_str[a], id_to_str[b]) for a, b, _ in tk["merges"]]
tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
tok.save(f"{PUB}/tokenizer.json")
assert tok.encode("Det var en mørk kveld i Oslo.").ids, "tokenizer rebuild failed"

SPAM = {
    "escort", "escorts", "eskorte", "eskortejenter", "shemale", "porno", "porn",
    "sexdating", "sexkontakt", "webcam", "webcams", "camgirls", "milf", "anal",
    "knulle", "pule", "kontaktannonser", "swingers", "chatroulette", "dogging",
    "massasjejenter", "thaimassasje", "sexleketøy", "dildo", "vibrator",
    "casino", "spilleautomater", "freespins", "gratisspinn", "bookmakere",
}

def clean_doc(t):
    t = t.strip()
    if len(t) < 200:
        return None
    keep = sum(1 for ch in t if ch.isalpha() or ch in " .,;:!?()-'\"\n")
    if keep / len(t) < 0.85:
        return None
    words = re.findall(r"[^\W\d_]+", t.lower(), re.UNICODE)
    if not words:
        return None
    # spam vocab: two strikes and the document is out
    if sum(1 for w in words if w in SPAM) >= 2:
        return None
    # keyword stuffing: no real prose repeats one word this hard
    top = collections.Counter(words).most_common(1)[0][1]
    if len(words) >= 50 and top / len(words) > 0.08:
        return None
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    return "".join(" " if (ch != "\n" and ch.isspace()) else ch for ch in t)

url = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/nob_Latn/train/000_00000.parquet"
f = fsspec.open(url, "rb").open()
pf = pq.ParquetFile(f)
print(f"{pf.num_row_groups} row groups", flush=True)

docs, total = [], 0
for rg in range(pf.num_row_groups):
    tbl = pf.read_row_group(rg, columns=["text"])
    for d in tbl["text"].to_pylist():
        d = clean_doc(d)
        if not d:
            continue
        docs.append(d)
        total += len(d)
    print(f"rg {rg}: {len(docs)} docs, {total/1e6:.0f}M chars", flush=True)
    if total >= TARGET_CHARS + VAL_CHARS:
        break

import random
random.Random(1337).shuffle(docs)

# words list for the real-words stat (from a sample, capped)
wc = collections.Counter()
for d in docs[:60000]:
    wc.update(re.findall(r"[^\W\d_]{2,}", d.lower(), re.UNICODE))
words = [w for w, c in wc.most_common(80000) if c >= 5]
json.dump(words, open(f"{PUB}/words-nob-g{GEN}.json", "w"))
print(f"words {len(words)}", flush=True)

# split off val docs, then tokenize both streams in batches
val_docs, train_docs, acc = [], [], 0
for d in docs:
    if acc < VAL_CHARS:
        val_docs.append(d)
        acc += len(d)
    else:
        train_docs.append(d)

def tokenize_stream(doc_list):
    out = []
    for i in range(0, len(doc_list), 2048):
        encs = tok.encode_batch([d + "\n\n" for d in doc_list[i : i + 2048]])
        for e in encs:
            out.extend(e.ids)
    return np.array(out, dtype=np.uint16)

val = tokenize_stream(val_docs)
val.tofile(f"{PUB}/tokens-nob-g{GEN}-val.bin")
print(f"val {len(val)/1e6:.2f}M tokens", flush=True)

train = tokenize_stream(train_docs)
print(f"train {len(train)/1e6:.1f}M tokens ({sum(len(d) for d in train_docs)/len(train):.2f} chars/tok)", flush=True)

shards = []
counts = []
for i in range(0, len(train), SHARD_TOKENS):
    name = f"tokens-nob-g{GEN}-{len(shards):03d}.bin"
    chunk = train[i : i + SHARD_TOKENS]
    chunk.tofile(f"{PUB}/{name}")
    shards.append(name)
    counts.append(len(chunk))
json.dump({"shards": shards, "counts": counts, "total": int(len(train)),
           "val": f"tokens-nob-g{GEN}-val.bin", "tok": f"tok16k-g{GEN}.json",
           "words": f"words-nob-g{GEN}.json", "gen": GEN},
          open(f"{PUB}/tokens-nob-meta.json", "w"))
print(f"DONE: {len(shards)} shards, {len(train)/1e6:.1f}M train tokens", flush=True)
