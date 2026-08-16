// Long-horizon training stability check at the production config on real corpus data.
// Run: deno run --unstable-webgpu --unstable-sloppy-imports --allow-read tests/drift_repro.ts
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
const cfg = { B: 16, T: 128, C: 384, L: 10, NH: 8, V: 16384, lr: 6e-4, warmup: 300, f16: hasF16 };
const model = new GPT(device, cfg);
const BT = cfg.B * cfg.T;

const t0 = new Uint16Array((await Deno.readFile('public/tokens-nob-000.bin')).buffer);
const t1 = new Uint16Array((await Deno.readFile('public/tokens-nob-001.bin')).buffer);
const toks = new Uint16Array(t0.length + t1.length);
toks.set(t0, 0); toks.set(t1, t0.length);
const valToks = new Uint16Array((await Deno.readFile('public/tokens-nob-val.bin')).buffer);

const batch = (src: Uint16Array) => {
  const x = new Uint32Array(BT);
  const y = new Uint32Array(BT);
  for (let b = 0; b < cfg.B; b++) {
    const off = Math.floor(Math.random() * (src.length - cfg.T - 1));
    for (let t = 0; t < cfg.T; t++) { x[b * cfg.T + t] = src[off + t]; y[b * cfg.T + t] = src[off + t + 1]; }
  }
  return { x, y };
};

let acc = 0, n = 0;
for (let i = 0; i < 15000; i++) {
  const { x, y } = batch(toks);
  const l = await model.trainStep(x, y, i % 2 === 1);
  if (isFinite(l)) { acc += l; n++; }
  if (i > 0 && i % 250 === 0) {
    const vb = batch(valToks);
    const v = await model.valStep(vb.x, vb.y);
    console.log(`step ${i} train ${(acc / Math.max(1, n)).toFixed(4)} val ${v.toFixed(4)}`);
    acc = 0; n = 0;
  }
}
console.log('DONE');
