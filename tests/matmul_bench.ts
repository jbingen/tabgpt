// Matmul kernel shootout on the real GPU. Run:
// deno run --unstable-webgpu --unstable-sloppy-imports tests/matmul_bench.ts

import * as K from '../src/engine/kernels';

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter!.requestDevice();
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

function dispatch(code: string, bufs: GPUBuffer[], wgx: number, wgy = 1, times = 1) {
  const module = device.createShaderModule({ code });
  const p = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
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

const ceil = (a: number, b: number) => Math.ceil(a / b);
const rand = (n: number) => Float32Array.from({ length: n }, () => (Math.random() - 0.5) * 2);

// vec4 columns: each thread computes 1 row x vec4 of columns
const vec4Fwd = (M: number, Kd: number, N: number) => `
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> bias_: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  let i = gid.y;
  if (j >= ${N / 4}u || i >= ${M}u) { return; }
  var acc = bias_[j];
  for (var k = 0u; k < ${Kd}u; k = k + 1u) {
    acc = acc + a[i * ${Kd}u + k] * b[k * ${N / 4}u + j];
  }
  out[i * ${N / 4}u + j] = acc;
}`;

// vec4 columns + R-row register blocking: each thread computes R rows x vec4 cols
const vec4RowsFwd = (M: number, Kd: number, N: number, R: number) => `
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> bias_: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  let i0 = gid.y * ${R}u;
  if (j >= ${N / 4}u || i0 >= ${M}u) { return; }
${Array.from({ length: R }, (_, r) => `  var acc${r} = bias_[j];`).join('\n')}
  for (var k = 0u; k < ${Kd}u; k = k + 1u) {
    let bv = b[k * ${N / 4}u + j];
${Array.from({ length: R }, (_, r) => `    acc${r} = acc${r} + a[(i0 + ${r}u) * ${Kd}u + k] * bv;`).join('\n')}
  }
${Array.from({ length: R }, (_, r) => `  if (i0 + ${r}u < ${M}u) { out[(i0 + ${r}u) * ${N / 4}u + j] = acc${r}; }`).join('\n')}
}`;

// vec4 rows + vec4 over k as well: a read as vec4 (needs K % 4 == 0)
const vec4KFwd = (M: number, Kd: number, N: number, R: number) => `
@group(0) @binding(0) var<storage, read_write> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> bias_: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  let i0 = gid.y * ${R}u;
  if (j >= ${N / 4}u || i0 >= ${M}u) { return; }
${Array.from({ length: R }, (_, r) => `  var acc${r} = bias_[j];`).join('\n')}
  for (var k4 = 0u; k4 < ${Kd / 4}u; k4 = k4 + 1u) {
    let b0 = b[(k4 * 4u + 0u) * ${N / 4}u + j];
    let b1 = b[(k4 * 4u + 1u) * ${N / 4}u + j];
    let b2 = b[(k4 * 4u + 2u) * ${N / 4}u + j];
    let b3 = b[(k4 * 4u + 3u) * ${N / 4}u + j];
${Array.from({ length: R }, (_, r) => `    {
      let av = a[(i0 + ${r}u) * ${Kd / 4}u + k4];
      acc${r} = acc${r} + av.x * b0 + av.y * b1 + av.z * b2 + av.w * b3;
    }`).join('\n')}
  }
${Array.from({ length: R }, (_, r) => `  if (i0 + ${r}u < ${M}u) { out[(i0 + ${r}u) * ${N / 4}u + j] = acc${r}; }`).join('\n')}
}`;

async function bench(name: string, code: string, bufs: GPUBuffer[], wgx: number, wgy: number, flops: number) {
  dispatch(code, bufs, wgx, wgy, 5);
  await read(bufs[bufs.length - 1], 1);
  const iters = 50;
  const t0 = performance.now();
  dispatch(code, bufs, wgx, wgy, iters);
  await read(bufs[bufs.length - 1], 1);
  const dt = (performance.now() - t0) / 1000;
  console.log(`${name}: ${((flops * iters) / dt / 1e9).toFixed(1)} GFLOPS`);
}

const M = 1024, Kd = 512, N = 512;
const aData = rand(M * Kd), wData = rand(Kd * N), biasData = rand(N);
const a = buf(aData), w = buf(wData), bias = buf(biasData), out = buf(new Float32Array(M * N));
const flops = 2 * M * Kd * N;

// correctness spot-check for the fastest candidates against naive
const want = new Float32Array(M * N);
{
  dispatch(K.matmulNaive(M, Kd, N, true), [a, w, bias, out], ceil(N, 16), ceil(M, 16));
  want.set(await read(out, M * N));
}
async function verify(name: string, code: string, wgx: number, wgy: number) {
  device.queue.writeBuffer(out, 0, new Float32Array(M * N));
  dispatch(code, [a, w, bias, out], wgx, wgy);
  const got = await read(out, M * N);
  let worst = 0;
  for (let i = 0; i < want.length; i++) worst = Math.max(worst, Math.abs(got[i] - want[i]) / (1 + Math.abs(want[i])));
  console.log(`${worst < 2e-3 ? 'PASS' : 'FAIL'} ${name} vs naive (err ${worst.toExponential(1)})`);
}

await bench('naive           ', K.matmulNaive(M, Kd, N, true), [a, w, bias, out], ceil(N, 16), ceil(M, 16), flops);
await bench('vec4            ', vec4Fwd(M, Kd, N), [a, w, bias, out], ceil(N / 4, 16), ceil(M, 16), flops);
for (const R of [2, 4, 8]) {
  await bench(`vec4 rows R=${R}    `, vec4RowsFwd(M, Kd, N, R), [a, w, bias, out], ceil(N / 4, 16), ceil(M / R, 16), flops);
}
for (const R of [2, 4, 8]) {
  await bench(`vec4K rows R=${R}   `, vec4KFwd(M, Kd, N, R), [a, w, bias, out], ceil(N / 4, 16), ceil(M / R, 16), flops);
}
await verify('vec4 rows R=4', vec4RowsFwd(M, Kd, N, 4), ceil(N / 4, 16), ceil(M / 4, 16));
await verify('vec4K rows R=4', vec4KFwd(M, Kd, N, 4), ceil(N / 4, 16), ceil(M / 4, 16));
