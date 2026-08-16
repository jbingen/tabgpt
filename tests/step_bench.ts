// Whole training-step benchmark at the page's real config.
// Run: deno run --unstable-webgpu --unstable-sloppy-imports tests/step_bench.ts

import { GPT } from '../src/engine/model';
import { setF16 } from '../src/engine/kernels';

const adapter = await navigator.gpu.requestAdapter();
const hasF16 = adapter!.features.has('shader-f16');
setF16(hasF16);
console.log(`f16: ${hasF16}`);
const device = await adapter!.requestDevice({
  requiredFeatures: hasF16 ? ['shader-f16' as GPUFeatureName] : [],
  requiredLimits: {
    maxBufferSize: adapter!.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter!.limits.maxStorageBufferBindingSize,
  },
});

const cfg = { B: 16, T: 128, C: 384, L: 10, NH: 8, V: 16384, lr: 1e-3, warmup: 200, f16: hasF16 };
const model = new GPT(device, cfg);
const BT = cfg.B * cfg.T;
const x = new Uint32Array(BT);
const y = new Uint32Array(BT);
for (let i = 0; i < BT; i++) { x[i] = i % cfg.V; y[i] = (i + 1) % cfg.V; }

for (let i = 0; i < 30; i++) await model.trainStep(x, y, i % 2 === 0); // warmup

const N = 200;
const t0 = performance.now();
for (let i = 0; i < N; i++) await model.trainStep(x, y, i % 2 === 1);
await model.trainStep(x, y, true); // final sync
const dt = (performance.now() - t0) / 1000;
console.log(`step ${((dt / (N + 1)) * 1000).toFixed(2)} ms, ${Math.round(((N + 1) * BT) / dt).toLocaleString()} tok/s`);
