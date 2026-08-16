// Headless kernel tests + benchmarks on the real GPU via deno's WebGPU.
// Run: deno run --unstable-webgpu --unstable-sloppy-imports tests/kernels_test.ts

import * as K from '../src/engine/kernels';

// unit tests exercise the f32 variants (CPU reference data is f32)
K.setF16(false);
import { GPT } from '../src/engine/model';

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error('no adapter');
const device = await adapter.requestDevice();

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

function buf(data: Float32Array): GPUBuffer {
  const b = device.createBuffer({ size: data.byteLength, usage: STORAGE });
  device.queue.writeBuffer(b, 0, data);
  return b;
}

async function read(b: GPUBuffer, numel: number): Promise<Float32Array> {
  const staging = device.createBuffer({ size: numel * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(b, 0, staging, 0, numel * 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(staging.getMappedRange()).slice();
  staging.unmap();
  staging.destroy();
  return out;
}

function pipeline(code: string) {
  const module = device.createShaderModule({ code });
  return device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
}

function dispatch(code: string, bufs: GPUBuffer[], wgx: number, wgy = 1, times = 1) {
  const p = pipeline(code);
  const bind = device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  for (let i = 0; i < times; i++) {
    pass.setPipeline(p);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(wgx, wgy);
  }
  pass.end();
  device.queue.submit([enc.finish()]);
}

const rand = (n: number) => Float32Array.from({ length: n }, () => (Math.random() - 0.5) * 2);
const ceil = (a: number, b: number) => Math.ceil(a / b);

let failed = 0;
function check(name: string, got: Float32Array, want: Float32Array, tol = 2e-3) {
  let worst = 0;
  for (let i = 0; i < want.length; i++) {
    const d = Math.abs(got[i] - want[i]) / (1 + Math.abs(want[i]));
    if (d > worst) worst = d;
  }
  const ok = worst < tol;
  if (!ok) failed++;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name} (max rel err ${worst.toExponential(2)})`);
}

// ---- tiled matmul forward vs CPU, odd sizes to exercise the bounds checks ----
{
  const M = 50, Kd = 36, N = 44;
  const a = rand(M * Kd), w = rand(Kd * N), bias = rand(N);
  const want = new Float32Array(M * N);
  for (let m = 0; m < M; m++)
    for (let n = 0; n < N; n++) {
      let s = bias[n];
      for (let k = 0; k < Kd; k++) s += a[m * Kd + k] * w[k * N + n];
      want[m * N + n] = s;
    }
  const out = buf(new Float32Array(M * N));
  dispatch(K.matmulFwd(M, Kd, N, true), [buf(a), buf(w), buf(bias), out], ceil(N, 64), ceil(M, 64));
  check('matmul fwd (tiled, bias)', await read(out, M * N), want);
}

// ---- backward dinp: dinp[M,K] += dout @ w^T ----
{
  const M = 37, Kd = 28, N = 52;
  const dout = rand(M * N), w = rand(Kd * N), init = rand(M * Kd);
  const want = Float32Array.from(init);
  for (let m = 0; m < M; m++)
    for (let k = 0; k < Kd; k++) {
      let s = 0;
      for (let n = 0; n < N; n++) s += dout[m * N + n] * w[k * N + n];
      want[m * Kd + k] += s;
    }
  const out = buf(Float32Array.from(init));
  dispatch(K.matmulBwdDinp(M, Kd, N), [buf(dout), buf(w), out], ceil(Kd, 64), ceil(M, 64));
  check('matmul bwd dinp (tiled, accum)', await read(out, M * Kd), want);
}

// ---- backward dw: dw[K,N] += inp^T @ dout ----
{
  const M = 41, Kd = 28, N = 36;
  const inp = rand(M * Kd), dout = rand(M * N), init = rand(Kd * N);
  const want = Float32Array.from(init);
  for (let k = 0; k < Kd; k++)
    for (let n = 0; n < N; n++) {
      let s = 0;
      for (let m = 0; m < M; m++) s += inp[m * Kd + k] * dout[m * N + n];
      want[k * N + n] += s;
    }
  const out = buf(Float32Array.from(init));
  dispatch(K.matmulBwdDw(M, Kd, N), [buf(inp), buf(dout), out], ceil(N, 64), ceil(Kd, 64));
  check('matmul bwd dw (tiled, accum)', await read(out, Kd * N), want);
}

// ---- benchmark naive vs tiled ----
async function bench(name: string, code: string, bufs: GPUBuffer[], wgx: number, wgy: number, flops: number) {
  dispatch(code, bufs, wgx, wgy, 5); // warmup
  await read(bufs[bufs.length - 1], 1);
  const iters = 50;
  const t0 = performance.now();
  dispatch(code, bufs, wgx, wgy, iters);
  await read(bufs[bufs.length - 1], 1); // forces completion
  const dt = (performance.now() - t0) / 1000;
  const gflops = (flops * iters) / dt / 1e9;
  console.log(`BENCH ${name}: ${gflops.toFixed(1)} GFLOPS`);
  return gflops;
}

{
  const M = 1024, Kd = 512, N = 512;
  const a = buf(rand(M * Kd)), w = buf(rand(Kd * N)), bias = buf(rand(N)), out = buf(new Float32Array(M * N));
  const flops = 2 * M * Kd * N;
  const naive = await bench('matmul naive 1024x512x512', K.matmulNaive(M, Kd, N, true), [a, w, bias, out], ceil(N, 16), ceil(M, 16), flops);
  const v4 = await bench('matmul vec4 1024x512x512', K.matmulFwd(M, Kd, N, true), [a, w, bias, out], ceil(N, 64), ceil(M, 64), flops);
  console.log(`BENCH speedup vec4 vs naive: ${(v4 / naive).toFixed(2)}x`);
}

// ---- end to end: a tiny GPT must learn a periodic token pattern ----
// runs on the f16 path when the adapter supports it, same as the real page
{
  const hasF16 = adapter.features.has('shader-f16');
  K.setF16(hasF16);
  console.log(`e2e f16: ${hasF16}`);
  const dev2 = await adapter.requestDevice({
    requiredFeatures: hasF16 ? ['shader-f16' as GPUFeatureName] : [],
  });
  const cfg = { B: 8, T: 32, C: 64, L: 2, NH: 2, V: 32, lr: 3e-3, warmup: 10, f16: hasF16 };
  const model = new GPT(dev2, cfg);
  const x = new Uint32Array(cfg.B * cfg.T);
  const y = new Uint32Array(cfg.B * cfg.T);
  const lossAt: number[] = [];
  for (let step = 0; step < 80; step++) {
    for (let b = 0; b < cfg.B; b++) {
      const off = Math.floor(Math.random() * 100);
      for (let t = 0; t < cfg.T; t++) {
        x[b * cfg.T + t] = (off + t) % 7;
        y[b * cfg.T + t] = (off + t + 1) % 7;
      }
    }
    lossAt.push(await model.trainStep(x, y));
  }
  const first = lossAt[0];
  const last = lossAt.slice(-5).reduce((a, b) => a + b) / 5;
  const ok = isFinite(last) && last < first - 1.5;
  if (!ok) failed++;
  console.log(`${ok ? 'PASS' : 'FAIL'} end-to-end training (loss ${first.toFixed(3)} -> ${last.toFixed(3)} in 80 steps)`);

  // valStep must agree with the training loss on the same distribution
  const vl = await model.valStep(x, y);
  const vOk = isFinite(vl) && Math.abs(vl - last) < 1.0;
  if (!vOk) failed++;
  console.log(`${vOk ? 'PASS' : 'FAIL'} valStep (${vl.toFixed(4)} vs train ${last.toFixed(4)})`);
}

console.log(failed === 0 ? 'ALL PASS' : `${failed} FAILURES`);
Deno.exit(failed === 0 ? 0 : 1);
