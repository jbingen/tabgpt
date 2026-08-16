# TabGPT

Pretrain a GPT in a browser tab. The full training loop (forward, backward, AdamW) is implemented as WebGPU compute shaders and runs on the local GPU with nothing installed. The page shows the live train/val loss, samples from the model as it learns, and exports the current weights as a standard .safetensors file at any point.

## Running

```sh
bun install
bun run dev
```

Requires a WebGPU-capable browser (Chrome, Edge, Arc, Safari 18+). Devices with the `shader-f16` feature train in mixed precision; others fall back to f32.

## Architecture

The model is a 24M parameter GPT: 10 layers, width 384, 8 heads, 128 tokens of context, 16k byte-level BPE vocabulary. It uses RMSNorm, rotary position embeddings, QK-norm, ReLU² MLPs, zero-initialized projections, tied embeddings and no bias terms, largely following [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt). Training data is a filtered 103M-token slice of the Bokmål portion of [FineWeb 2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), tokenized offline and served as binary shards with a held-out validation set.

## Implementation

- `src/engine/kernels.ts`: every op as hand-written WGSL, forward and backward: embeddings, RoPE, attention, matmuls, RMSNorm, QK-norm, ReLU², softmax cross-entropy, global gradient-norm clipping, AdamW. Shapes are baked at pipeline creation, so a training step is a fixed prebuilt dispatch list in the spirit of [llm.c](https://github.com/karpathy/llm.c).
- `src/engine/model.ts`: the training graph. Parameters, gradients and optimizer state live in pooled buffers (two `clearBuffer` calls and a three-dispatch optimizer per step). Mixed precision keeps f32 master weights and Adam moments with an f16 shadow for compute. Command buffers are kept short so the window compositor is never starved.
- `src/engine/data.ts`: progressive shard loading, byte-level BPE encode/decode in the browser.
- `src/main.ts`: the training loop and page. Training state (weights and optimizer moments) checkpoints to IndexedDB and resumes exactly across reloads.

The matmuls are vec4 register-blocked and reach roughly 1 TFLOPS f32 on Apple silicon (shared-memory tiling loses to this shape on Apple GPUs). Attention is not yet fused; that and longer context via flash attention are the main open performance work.

## Tests

```sh
bun run test    # kernel correctness vs CPU references, end-to-end learning check
bun run bench   # matmul kernel comparison
```

Both run headlessly on the real GPU via Deno's WebGPU implementation.

## License

MIT
