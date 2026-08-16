// valStep parity check against the training loss at production scale.
// Run: deno run --unstable-webgpu --unstable-sloppy-imports tests/val_repro.ts
import { GPT } from '../src/engine/model';
import { setF16 } from '../src/engine/kernels';

const adapter = await navigator.gpu.requestAdapter();
const hasF16 = adapter!.features.has('shader-f16');
setF16(hasF16);
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

// real corpus tokens so the distribution matches the page
const bin = await Deno.readFile('public/tokens-nob-000.bin');
const toks = new Uint16Array(bin.buffer, 0, 4_000_000);
const batch = () => {
  const x = new Uint32Array(BT);
  const y = new Uint32Array(BT);
  for (let b = 0; b < cfg.B; b++) {
    const off = Math.floor(Math.random() * (toks.length - cfg.T - 1));
    for (let t = 0; t < cfg.T; t++) { x[b * cfg.T + t] = toks[off + t]; y[b * cfg.T + t] = toks[off + t + 1]; }
  }
  return { x, y };
};

for (let i = 0; i < 60; i++) {
  const { x, y } = batch();
  const l = await model.trainStep(x, y, i % 2 === 1);
  if (i % 2 === 1 && (i < 4 || i > 56)) console.log(`train step ${i}: ${l.toFixed(4)}`);
}
for (let k = 0; k < 4; k++) {
  const { x, y } = batch();
  const v = await model.valStep(x, y);
  console.log(`valStep ${k}: ${v} ${isFinite(v) ? 'OK' : 'NON-FINITE'}`);
  if (!isFinite(v)) console.log('debug:', await model.debugVal());
}
