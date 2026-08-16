# Trains the 16k byte-level BPE tokenizer from a raw text corpus at
# public/corpus.txt and exports the browser-side vocab/merges tables.
import collections
import os
import json
import re

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PUB = os.path.join(os.path.dirname(__file__), "..", "public")
text = open(f"{PUB}/corpus.txt").read()

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=16384,
    min_frequency=2,
    show_progress=False,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
tok.train_from_iterator([text], trainer)

ids = tok.encode(text).ids
arr = np.array(ids, dtype=np.uint16)
arr.tofile(f"{PUB}/tokens-nob.bin")
print(f"{len(text)/1e6:.1f}M chars -> {len(ids)/1e6:.2f}M tokens ({len(text)/len(ids):.2f} chars/token)")

# GPT-2's byte<->unicode table, to turn byte-level token strings back into bytes
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

byte_decoder = {c: b for b, c in bytes_to_unicode().items()}
vocab = tok.get_vocab()
id_to_bytes = [None] * len(vocab)
for s, i in vocab.items():
    id_to_bytes[i] = [byte_decoder[ch] for ch in s]

tj = json.loads(tok.to_str())
merges = []
for m in tj["model"]["merges"]:
    a, b = (m if isinstance(m, list) else m.split(" ", 1))
    merges.append([vocab[a], vocab[b], vocab[a + b]])

json.dump({"vocab": id_to_bytes, "merges": merges}, open(f"{PUB}/tok16k.json", "w"))
print(f"vocab {len(vocab)}, merges {len(merges)}")

words = collections.Counter(re.findall(r"[^\W\d_]{2,}", text.lower(), re.UNICODE))
keep = [w for w, c in words.items() if c >= 3]
json.dump(keep, open(f"{PUB}/words-nob.json", "w"))
print(f"words {len(keep)}")
