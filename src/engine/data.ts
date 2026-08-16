export interface TokenData {
  tokens: Uint16Array;          // train stream; grows as shards arrive
  filled: number;               // how much of tokens is loaded so far
  val: Uint16Array;             // held-out documents, never trained on
  V: number;
  vocabBytes: Uint8Array[];     // token id -> utf8 bytes
  byteToId: Int32Array;         // raw byte -> base token id
  mergeRank: Map<number, { rank: number; id: number }>;
  words: Set<string>;           // corpus vocabulary, for the real-words stat
  gen: number;                  // corpus generation; bumping it retires models
}

async function fetchWithProgress(url: string, onBytes?: (got: number, total: number) => void): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok || !r.body || !onBytes) return r.arrayBuffer();
  const total = Number(r.headers.get('content-length') ?? 0);
  const reader = r.body.getReader();
  const chunks: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.byteLength;
    onBytes(got, total);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.byteLength; }
  return out.buffer;
}

export async function loadTokenData(base = '', onBytes?: (got: number, total: number) => void): Promise<TokenData> {
  const [tok, words] = await Promise.all([
    fetch(`${base}/tok16k.json`).then((r) => r.json()),
    fetch(`${base}/words-nob.json`).then((r) => r.json()),
  ]);
  const vocabBytes: Uint8Array[] = (tok.vocab as number[][]).map((a) => Uint8Array.from(a));
  const byteToId = new Int32Array(256).fill(-1);
  vocabBytes.forEach((b, id) => {
    if (b.length === 1) byteToId[b[0]] = id;
  });
  const mergeRank = new Map<number, { rank: number; id: number }>();
  (tok.merges as number[][]).forEach(([a, b, id], rank) => {
    mergeRank.set(a * 65536 + b, { rank, id });
  });
  const td: TokenData = {
    tokens: new Uint16Array(0), filled: 0, val: new Uint16Array(0),
    V: vocabBytes.length, vocabBytes, byteToId, mergeRank,
    words: new Set(words as string[]),
    gen: 1,
  };

  let meta: { shards: string[]; counts: number[]; total: number; val: string; gen?: number } | null = null;
  try {
    const r = await fetch(`${base}/tokens-nob-meta.json`);
    if (r.ok) meta = await r.json();
  } catch { /* fall back to the single-file corpus */ }

  if (!meta) {
    const bin = await (await fetch(`${base}/tokens-nob.bin`)).arrayBuffer();
    const u16 = new Uint16Array(bin);
    const split = Math.floor(u16.length * 0.95);
    td.tokens = u16.subarray(0, split);
    td.filled = split;
    td.val = u16.subarray(split);
    return td;
  }

  td.gen = meta.gen ?? 1;
  // sharded corpus: train on shard 0 immediately, stream the rest in the
  // background. Machines with little memory only take the first two shards.
  const mem = (navigator as any).deviceMemory ?? 8;
  const nShards = mem < 8 ? Math.min(2, meta.shards.length) : meta.shards.length;
  const total = meta.counts.slice(0, nShards).reduce((a, b) => a + b, 0);
  td.tokens = new Uint16Array(total);
  const [first, valBin] = await Promise.all([
    fetchWithProgress(`${base}/${meta.shards[0]}`, onBytes),
    fetch(`${base}/${meta.val}`).then((r) => r.arrayBuffer()),
  ]);
  td.tokens.set(new Uint16Array(first), 0);
  td.filled = meta.counts[0];
  td.val = new Uint16Array(valBin);
  (async () => {
    let off = meta.counts[0];
    for (let i = 1; i < nShards; i++) {
      try {
        const buf = await (await fetch(`${base}/${meta.shards[i]}`)).arrayBuffer();
        td.tokens.set(new Uint16Array(buf), off);
        off += meta.counts[i];
        td.filled = off;
        console.log(`tabgpt: corpus ${(td.filled / 1e6).toFixed(0)}M / ${(total / 1e6).toFixed(0)}M tokens`);
      } catch {
        break;
      }
    }
  })();
  return td;
}

export function decodeTokens(td: TokenData, ids: number[]): string {
  let total = 0;
  for (const id of ids) total += td.vocabBytes[id]?.length ?? 0;
  const bytes = new Uint8Array(total);
  let o = 0;
  for (const id of ids) {
    const b = td.vocabBytes[id];
    if (!b) continue;
    bytes.set(b, o);
    o += b.length;
  }
  return new TextDecoder().decode(bytes);
}

/** Byte-level BPE encode: start from byte tokens, apply merges by rank. */
export function encodeText(td: TokenData, s: string): number[] {
  const bytes = new TextEncoder().encode(s);
  const ids: number[] = [];
  for (const b of bytes) {
    const id = td.byteToId[b];
    if (id >= 0) ids.push(id);
  }
  while (ids.length > 1) {
    let best = -1;
    let bestRank = Infinity;
    let bestId = 0;
    for (let i = 0; i < ids.length - 1; i++) {
      const m = td.mergeRank.get(ids[i] * 65536 + ids[i + 1]);
      if (m && m.rank < bestRank) { bestRank = m.rank; best = i; bestId = m.id; }
    }
    if (best < 0) break;
    ids.splice(best, 2, bestId);
  }
  return ids;
}

export function sampleBatch(data: Uint16Array, limit: number, B: number, T: number, rng: () => number) {
  const x = new Uint32Array(B * T);
  const y = new Uint32Array(B * T);
  for (let b = 0; b < B; b++) {
    const off = Math.floor(rng() * (limit - T - 1));
    for (let t = 0; t < T; t++) {
      x[b * T + t] = data[off + t];
      y[b * T + t] = data[off + t + 1];
    }
  }
  return { x, y };
}
