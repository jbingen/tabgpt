// WGSL kernels for the full training step. Shapes are baked at pipeline
// creation; the only uniform is the optimizer's step counter and learning
// rate. With f16 enabled, weights (as a shadow copy), activations and
// gradients are stored half-precision with saturating casts; master weights,
// Adam moments, in-kernel accumulators and norm statistics stay f32.
// Backward kernels accumulate into gradient pools that are zeroed each step.

let F16 = false;
export const setF16 = (on: boolean) => { F16 = on; };
export const isF16 = () => F16;

const EN = () => (F16 ? 'enable f16;\n' : '');
const A4 = () => (F16 ? 'array<vec4<f16>>' : 'array<vec4<f32>>');
const A1 = () => (F16 ? 'array<f16>' : 'array<f32>');
const up4 = (x: string) => (F16 ? `vec4<f32>(${x})` : `(${x})`);
const up = (x: string) => (F16 ? `f32(${x})` : `(${x})`);
const dn4 = (x: string) => (F16 ? `vec4<f16>(clamp(${x}, vec4<f32>(-65504.0), vec4<f32>(65504.0)))` : `(${x})`);
const dn = (x: string) => (F16 ? `f16(clamp(${x}, -65504.0, 65504.0))` : `(${x})`);

const R4 = [0, 1, 2, 3];

// Token embedding lookup; position information comes from RoPE.
export const embedFwd = (B: number, T: number, C: number) => `${EN()}
@group(0) @binding(0) var<storage, read_write> ix: array<u32>;
@group(0) @binding(1) var<storage, read_write> wte: ${A1()};
@group(0) @binding(2) var<storage, read_write> out: ${A1()};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${B * T * C}u) { return; }
  out[i] = wte[ix[i / ${C}u] * ${C}u + (i % ${C}u)];
}`;

// One workgroup per row; rrms is saved for the backward pass.
export const rmsnormFwd = (N: number, C: number) => `${EN()}
@group(0) @binding(0) var<storage, read_write> inp: ${A1()};
@group(0) @binding(1) var<storage, read_write> w: ${A1()};
@group(0) @binding(2) var<storage, read_write> out: ${A1()};
@group(0) @binding(3) var<storage, read_write> rrmsB: array<f32>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let base = wid.x * ${C}u;
  var ss = 0.0;
  for (var c = lid.x; c < ${C}u; c = c + 128u) {
    let v = ${up('inp[base + c]')};
    ss = ss + v * v;
  }
  red[lid.x] = ss;
  workgroupBarrier();
  for (var off = 64u; off > 0u; off = off >> 1u) {
    if (lid.x < off) { red[lid.x] = red[lid.x] + red[lid.x + off]; }
    workgroupBarrier();
  }
  let rrms = 1.0 / sqrt(red[0] / ${C}.0 + 1e-5);
  for (var c = lid.x; c < ${C}u; c = c + 128u) {
    out[base + c] = ${dn(`${up('inp[base + c]')} * rrms * ${up('w[c]')}`)};
  }
  if (lid.x == 0u) { rrmsB[wid.x] = rrms; }
}`;

// rotary position embedding, applied in place to the q and k sections of qkv.
// invert=true rotates by -theta, which is the exact backward pass.
export const rope = (B: number, T: number, C: number, NH: number, invert: boolean) => {
  const hs = C / NH;
  const hp = hs / 2;
  return `${EN()}
@group(0) @binding(0) var<storage, read_write> buf: ${A1()};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${B * T * 2 * NH * hp}u) { return; }
  let j = i % ${hp}u;
  let h = (i / ${hp}u) % ${NH}u;
  let s = (i / ${hp * NH}u) % 2u;
  let bt = i / ${hp * NH * 2}u;
  let t = bt % ${T}u;
  let base = bt * ${3 * C}u + s * ${C}u + h * ${hs}u + 2u * j;
  let theta = pow(10000.0, -2.0 * f32(j) / ${hs}.0);
  let ang = ${invert ? '-' : ''}f32(t) * theta;
  let cw = cos(ang);
  let sw = sin(ang);
  let x = ${up('buf[base]')};
  let y = ${up('buf[base + 1u]')};
  buf[base] = ${dn('x * cw - y * sw')};
  buf[base + 1u] = ${dn('x * sw + y * cw')};
}`;
};

// QK-norm: unit-RMS normalization of each q and k head vector, in place.
// Bounds the attention logits, which otherwise grow without limit on long
// runs. Applied after RoPE; rotation preserves norms, so the order is
// mathematically irrelevant.
export const qknormFwd = (B: number, T: number, C: number, NH: number) => {
  const hs = C / NH;
  const R = B * T * 2 * NH;
  return `${EN()}
@group(0) @binding(0) var<storage, read_write> qkv: ${A1()};
@group(0) @binding(1) var<storage, read_write> rrmsB: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= ${R}u) { return; }
  let h = r % ${NH}u;
  let s = (r / ${NH}u) % 2u;
  let bt = r / ${2 * NH}u;
  let base = bt * ${3 * C}u + s * ${C}u + h * ${hs}u;
  var ss = 0.0;
  for (var i = 0u; i < ${hs}u; i = i + 1u) {
    let v = ${up('qkv[base + i]')};
    ss = ss + v * v;
  }
  let rrms = 1.0 / sqrt(ss / ${hs}.0 + 1e-6);
  for (var i = 0u; i < ${hs}u; i = i + 1u) {
    qkv[base + i] = ${dn(`${up('qkv[base + i]')} * rrms`)};
  }
  rrmsB[r] = rrms;
}`;
};

// backward of the unit-RMS norm, in place on dqkv. Reconstructs everything
// from the stored post-norm values (still in qkv) and the saved rrms.
export const qknormBwd = (B: number, T: number, C: number, NH: number) => {
  const hs = C / NH;
  const R = B * T * 2 * NH;
  return `${EN()}
@group(0) @binding(0) var<storage, read_write> qkv: ${A1()};
@group(0) @binding(1) var<storage, read_write> rrmsB: array<f32>;
@group(0) @binding(2) var<storage, read_write> dqkv: ${A1()};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= ${R}u) { return; }
  let h = r % ${NH}u;
  let s = (r / ${NH}u) % 2u;
  let bt = r / ${2 * NH}u;
  let base = bt * ${3 * C}u + s * ${C}u + h * ${hs}u;
  let rrms = rrmsB[r];
  var m = 0.0;
  for (var i = 0u; i < ${hs}u; i = i + 1u) {
    m = m + ${up('dqkv[base + i]')} * ${up('qkv[base + i]')};
  }
  m = m / ${hs}.0;
  for (var i = 0u; i < ${hs}u; i = i + 1u) {
    let dy = ${up('dqkv[base + i]')};
    let y = ${up('qkv[base + i]')};
    dqkv[base + i] = ${dn('(dy - y * m) * rrms')};
  }
}`;
};

// naive f32 reference matmul, kept only for the test harness.
// out[M,N] = inp[M,K] @ w[K,N]
export const matmulNaive = (M: number, K: number, N: number, bias: boolean) => `
@group(0) @binding(0) var<storage, read_write> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
${bias
  ? `@group(0) @binding(2) var<storage, read_write> bias_: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;`
  : `@group(0) @binding(2) var<storage, read_write> out: array<f32>;`}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  let m = gid.y;
  if (n >= ${N}u || m >= ${M}u) { return; }
  var acc = ${bias ? 'bias_[n]' : '0.0'};
  for (var k = 0u; k < ${K}u; k = k + 1u) {
    acc = acc + inp[m * ${K}u + k] * w[k * ${N}u + n];
  }
  out[m * ${N}u + n] = acc;
}`;

// The winning shape on Apple GPUs: buffers bound as vec4 arrays, each thread
// owns 4 output rows x one vec4 of columns, reduction unrolled 4 wide,
// f32 accumulators. Requires K % 4 == 0 and N % 4 == 0.
// Dispatch: workgroup_size(16,16), so x = ceil(cols/64), y = ceil(rows/64).

export const matmulFwd = (M: number, K: number, N: number, bias: boolean) => {
  if (K % 4 || N % 4) throw new Error(`matmulFwd dims must be multiples of 4, got K=${K} N=${N}`);
  const K4 = K / 4, N4 = N / 4;
  return `${EN()}
@group(0) @binding(0) var<storage, read_write> a: ${A4()};
@group(0) @binding(1) var<storage, read_write> b: ${A4()};
${bias
  ? `@group(0) @binding(2) var<storage, read_write> bias_: ${A4()};
@group(0) @binding(3) var<storage, read_write> out: ${A4()};`
  : `@group(0) @binding(2) var<storage, read_write> out: ${A4()};`}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  let i0 = gid.y * 4u;
  if (j >= ${N4}u || i0 >= ${M}u) { return; }
${R4.map((r) => `  var acc${r} = ${bias ? up4('bias_[j]') : 'vec4<f32>()'};`).join('\n')}
  for (var k4 = 0u; k4 < ${K4}u; k4 = k4 + 1u) {
    let b0 = ${up4(`b[(k4 * 4u + 0u) * ${N4}u + j]`)};
    let b1 = ${up4(`b[(k4 * 4u + 1u) * ${N4}u + j]`)};
    let b2 = ${up4(`b[(k4 * 4u + 2u) * ${N4}u + j]`)};
    let b3 = ${up4(`b[(k4 * 4u + 3u) * ${N4}u + j]`)};
${R4.map((r) => `    {
      let av = ${up4(`a[(i0 + ${r}u) * ${K4}u + k4]`)};
      acc${r} = acc${r} + av.x * b0 + av.y * b1 + av.z * b2 + av.w * b3;
    }`).join('\n')}
  }
${R4.map((r) => `  if (i0 + ${r}u < ${M}u) { out[(i0 + ${r}u) * ${N4}u + j] = ${dn4(`acc${r}`)}; }`).join('\n')}
}`;
};

// out[I,J] (+)= A[I,L] @ B[J,L]^T, both operands contiguous over L.
// Serves as matmul-backward-dinp AND the tied lm_head forward.
export const matmulABt = (I: number, J: number, L: number, accum: boolean) => {
  if (J % 4 || L % 4) throw new Error(`matmulABt dims must be multiples of 4, got J=${J} L=${L}`);
  const J4 = J / 4, L4 = L / 4;
  return `${EN()}
@group(0) @binding(0) var<storage, read_write> a: ${A4()};
@group(0) @binding(1) var<storage, read_write> b: ${A4()};
@group(0) @binding(2) var<storage, read_write> out: ${A4()};
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j4 = gid.x;
  let i0 = gid.y * 4u;
  if (j4 >= ${J4}u || i0 >= ${I}u) { return; }
${R4.map((r) => `  var acc${r} = vec4<f32>();`).join('\n')}
  for (var l4 = 0u; l4 < ${L4}u; l4 = l4 + 1u) {
    let b0 = ${up4(`b[(j4 * 4u + 0u) * ${L4}u + l4]`)};
    let b1 = ${up4(`b[(j4 * 4u + 1u) * ${L4}u + l4]`)};
    let b2 = ${up4(`b[(j4 * 4u + 2u) * ${L4}u + l4]`)};
    let b3 = ${up4(`b[(j4 * 4u + 3u) * ${L4}u + l4]`)};
${R4.map((r) => `    {
      let av = ${up4(`a[(i0 + ${r}u) * ${L4}u + l4]`)};
      acc${r} = acc${r} + vec4<f32>(dot(av, b0), dot(av, b1), dot(av, b2), dot(av, b3));
    }`).join('\n')}
  }
${R4.map((r) => `  if (i0 + ${r}u < ${I}u) {
    let o${r} = (i0 + ${r}u) * ${J4}u + j4;
    out[o${r}] = ${accum ? `out[o${r}] + ${dn4(`acc${r}`)}` : dn4(`acc${r}`)};
  }`).join('\n')}
}`;
};

// dinp[M,K] += dout[M,N] @ w^T  (w stored [K,N])
export const matmulBwdDinp = (M: number, K: number, N: number) => matmulABt(M, K, N, true);

// dw[K,N] += inp^T @ dout
export const matmulBwdDw = (M: number, K: number, N: number) => {
  if (K % 4 || N % 4) throw new Error(`matmulBwdDw dims must be multiples of 4, got K=${K} N=${N}`);
  const K4 = K / 4, N4 = N / 4;
  return `${EN()}
@group(0) @binding(0) var<storage, read_write> inp: ${A4()};
@group(0) @binding(1) var<storage, read_write> dout: ${A4()};
@group(0) @binding(2) var<storage, read_write> dw: ${A4()};
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n4 = gid.x;
  let k4 = gid.y;
  if (n4 >= ${N4}u || k4 >= ${K4}u) { return; }
${R4.map((r) => `  var acc${r} = vec4<f32>();`).join('\n')}
  for (var m = 0u; m < ${M}u; m = m + 1u) {
    let av = ${up4(`inp[m * ${K4}u + k4]`)};
    let dv = ${up4(`dout[m * ${N4}u + n4]`)};
    acc0 = acc0 + av.x * dv;
    acc1 = acc1 + av.y * dv;
    acc2 = acc2 + av.z * dv;
    acc3 = acc3 + av.w * dv;
  }
${R4.map((r) => `  {
    let o${r} = (k4 * 4u + ${r}u) * ${N4}u + n4;
    dw[o${r}] = dw[o${r}] + ${dn4(`acc${r}`)};
  }`).join('\n')}
}`;
};

export const attnScores = (B: number, T: number, C: number, NH: number) => {
  const hs = C / NH;
  const hs4 = hs / 4;
  const C4 = C / 4;
  const scale = 1.0 / Math.sqrt(hs);
  return `${EN()}
@group(0) @binding(0) var<storage, read_write> qkv: ${A4()};
@group(0) @binding(1) var<storage, read_write> preatt: ${A1()};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${B * NH * T * T}u) { return; }
  let t2 = i % ${T}u;
  let t = (i / ${T}u) % ${T}u;
  if (t2 > t) { preatt[i] = ${dn('0.0')}; return; }
  let h = (i / ${T * T}u) % ${NH}u;
  let b = i / ${T * T * NH}u;
  let qoff = (b * ${T}u + t) * ${3 * C4}u + h * ${hs4}u;
  let koff = (b * ${T}u + t2) * ${3 * C4}u + ${C4}u + h * ${hs4}u;
  var acc = 0.0;
  for (var k = 0u; k < ${hs4}u; k = k + 1u) {
    acc = acc + dot(${up4('qkv[qoff + k]')}, ${up4('qkv[koff + k]')});
  }
  preatt[i] = ${dn(`acc * ${scale}`)};
}`;
};

// One workgroup per (b,h,t) row, T threads; T must be a power of two <= 256.
// T must be a power of two <= 256.
export const attnSoftmax = (B: number, T: number, NH: number) => `${EN()}
@group(0) @binding(0) var<storage, read_write> preatt: ${A1()};
@group(0) @binding(1) var<storage, read_write> att: ${A1()};
var<workgroup> red: array<f32, ${T}>;
@compute @workgroup_size(${T})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x % ${T}u;
  let base = wid.x * ${T}u;
  let j = lid.x;
  let val = ${up('preatt[base + j]')};
  red[j] = select(-1e30, val, j <= t);
  workgroupBarrier();
  for (var off = ${T / 2}u; off > 0u; off = off >> 1u) {
    if (j < off) { red[j] = max(red[j], red[j + off]); }
    workgroupBarrier();
  }
  let mx = red[0];
  workgroupBarrier();
  let e = select(0.0, exp(val - mx), j <= t);
  red[j] = e;
  workgroupBarrier();
  for (var off = ${T / 2}u; off > 0u; off = off >> 1u) {
    if (j < off) { red[j] = red[j] + red[j + off]; }
    workgroupBarrier();
  }
  att[base + j] = ${dn('e / red[0]')};
}`;

export const attnAgg = (B: number, T: number, C: number, NH: number) => {
  const hs4 = C / NH / 4;
  const C4 = C / 4;
  return `${EN()}
@group(0) @binding(0) var<storage, read_write> att: ${A1()};
@group(0) @binding(1) var<storage, read_write> qkv: ${A4()};
@group(0) @binding(2) var<storage, read_write> atty: ${A4()};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${B * T * C4}u) { return; }
  let c4 = i % ${C4}u;
  let t = (i / ${C4}u) % ${T}u;
  let b = i / ${C4 * T}u;
  let h = c4 / ${hs4}u;
  let attbase = ((b * ${NH}u + h) * ${T}u + t) * ${T}u;
  var acc = vec4<f32>();
  for (var t2 = 0u; t2 <= t; t2 = t2 + 1u) {
    acc = acc + ${up('att[attbase + t2]')} * ${up4(`qkv[(b * ${T}u + t2) * ${3 * C4}u + ${2 * C4}u + c4]`)};
  }
  atty[i] = ${dn4('acc')};
}`;
};

export const attnAggBwdDatt = (B: number, T: number, C: number, NH: number) => {
  const hs4 = C / NH / 4;
  const C4 = C / 4;
  return `${EN()}
@group(0) @binding(0) var<storage, read_write> datty: ${A4()};
@group(0) @binding(1) var<storage, read_write> qkv: ${A4()};
@group(0) @binding(2) var<storage, read_write> datt: ${A1()};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${B * NH * T * T}u) { return; }
  let t2 = i % ${T}u;
  let t = (i / ${T}u) % ${T}u;
  if (t2 > t) { return; }
  let h = (i / ${T * T}u) % ${NH}u;
  let b = i / ${T * T * NH}u;
  var acc = 0.0;
  for (var k = 0u; k < ${hs4}u; k = k + 1u) {
    acc = acc + dot(${up4(`datty[(b * ${T}u + t) * ${C4}u + h * ${hs4}u + k]`)},
                    ${up4(`qkv[(b * ${T}u + t2) * ${3 * C4}u + ${2 * C4}u + h * ${hs4}u + k]`)});
  }
  datt[i] = datt[i] + ${dn('acc')};
}`;
};

export const attnAggBwdDv = (B: number, T: number, C: number, NH: number) => {
  const hs4 = C / NH / 4;
  const C4 = C / 4;
  return `${EN()}
@group(0) @binding(0) var<storage, read_write> datty: ${A4()};
@group(0) @binding(1) var<storage, read_write> att: ${A1()};
@group(0) @binding(2) var<storage, read_write> dqkv: ${A4()};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${B * T * C4}u) { return; }
  let c4 = i % ${C4}u;
  let t2 = (i / ${C4}u) % ${T}u;
  let b = i / ${C4 * T}u;
  let h = c4 / ${hs4}u;
  var acc = vec4<f32>();
  for (var t = t2; t < ${T}u; t = t + 1u) {
    acc = acc + ${up(`att[((b * ${NH}u + h) * ${T}u + t) * ${T}u + t2]`)}
               * ${up4(`datty[(b * ${T}u + t) * ${C4}u + c4]`)};
  }
  let o = (b * ${T}u + t2) * ${3 * C4}u + ${2 * C4}u + c4;
  dqkv[o] = dqkv[o] + ${dn4('acc')};
}`;
};

// One workgroup per (b,h,t) row.
export const attnSoftmaxBwd = (B: number, T: number, NH: number) => `${EN()}
@group(0) @binding(0) var<storage, read_write> datt: ${A1()};
@group(0) @binding(1) var<storage, read_write> att: ${A1()};
@group(0) @binding(2) var<storage, read_write> dpreatt: ${A1()};
var<workgroup> red: array<f32, ${T}>;
@compute @workgroup_size(${T})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = wid.x % ${T}u;
  let base = wid.x * ${T}u;
  let j = lid.x;
  let a = ${up('att[base + j]')};
  let d = ${up('datt[base + j]')};
  red[j] = select(0.0, a * d, j <= t);
  workgroupBarrier();
  for (var off = ${T / 2}u; off > 0u; off = off >> 1u) {
    if (j < off) { red[j] = red[j] + red[j + off]; }
    workgroupBarrier();
  }
  if (j <= t) { dpreatt[base + j] = dpreatt[base + j] + ${dn('a * (d - red[0])')}; }
}`;

export const attnScoresBwdDq = (B: number, T: number, C: number, NH: number) => {
  const hs4 = C / NH / 4;
  const C4 = C / 4;
  const scale = 1.0 / Math.sqrt(C / NH);
  return `${EN()}
@group(0) @binding(0) var<storage, read_write> dpreatt: ${A1()};
@group(0) @binding(1) var<storage, read_write> qkv: ${A4()};
@group(0) @binding(2) var<storage, read_write> dqkv: ${A4()};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${B * T * C4}u) { return; }
  let c4 = i % ${C4}u;
  let t = (i / ${C4}u) % ${T}u;
  let b = i / ${C4 * T}u;
  let h = c4 / ${hs4}u;
  var acc = vec4<f32>();
  for (var t2 = 0u; t2 <= t; t2 = t2 + 1u) {
    acc = acc + ${up(`dpreatt[((b * ${NH}u + h) * ${T}u + t) * ${T}u + t2]`)}
               * ${up4(`qkv[(b * ${T}u + t2) * ${3 * C4}u + ${C4}u + c4]`)};
  }
  let o = (b * ${T}u + t) * ${3 * C4}u + c4;
  dqkv[o] = dqkv[o] + ${dn4(`acc * ${scale}`)};
}`;
};

export const attnScoresBwdDk = (B: number, T: number, C: number, NH: number) => {
  const hs4 = C / NH / 4;
  const C4 = C / 4;
  const scale = 1.0 / Math.sqrt(C / NH);
  return `${EN()}
@group(0) @binding(0) var<storage, read_write> dpreatt: ${A1()};
@group(0) @binding(1) var<storage, read_write> qkv: ${A4()};
@group(0) @binding(2) var<storage, read_write> dqkv: ${A4()};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${B * T * C4}u) { return; }
  let c4 = i % ${C4}u;
  let t2 = (i / ${C4}u) % ${T}u;
  let b = i / ${C4 * T}u;
  let h = c4 / ${hs4}u;
  var acc = vec4<f32>();
  for (var t = t2; t < ${T}u; t = t + 1u) {
    acc = acc + ${up(`dpreatt[((b * ${NH}u + h) * ${T}u + t) * ${T}u + t2]`)}
               * ${up4(`qkv[(b * ${T}u + t) * ${3 * C4}u + c4]`)};
  }
  let o = (b * ${T}u + t2) * ${3 * C4}u + ${C4}u + c4;
  dqkv[o] = dqkv[o] + ${dn4(`acc * ${scale}`)};
}`;
};

export const residualFwd = (N: number) => `${EN()}
@group(0) @binding(0) var<storage, read_write> a: ${A1()};
@group(0) @binding(1) var<storage, read_write> b: ${A1()};
@group(0) @binding(2) var<storage, read_write> out: ${A1()};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${N}u) { return; }
  out[i] = ${dn(`${up('a[i]')} + ${up('b[i]')}`)};
}`;

export const residualBwd = (N: number) => `${EN()}
@group(0) @binding(0) var<storage, read_write> dout: ${A1()};
@group(0) @binding(1) var<storage, read_write> da: ${A1()};
@group(0) @binding(2) var<storage, read_write> db: ${A1()};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${N}u) { return; }
  da[i] = ${dn(`${up('da[i]')} + ${up('dout[i]')}`)};
  db[i] = ${dn(`${up('db[i]')} + ${up('dout[i]')}`)};
}`;

// relu(x)^2
export const relu2Fwd = (N: number) => `${EN()}
@group(0) @binding(0) var<storage, read_write> inp: ${A1()};
@group(0) @binding(1) var<storage, read_write> out: ${A1()};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${N}u) { return; }
  let x = max(${up('inp[i]')}, 0.0);
  out[i] = ${dn('x * x')};
}`;

export const relu2Bwd = (N: number) => `${EN()}
@group(0) @binding(0) var<storage, read_write> dout: ${A1()};
@group(0) @binding(1) var<storage, read_write> inp: ${A1()};
@group(0) @binding(2) var<storage, read_write> dinp: ${A1()};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${N}u) { return; }
  dinp[i] = dinp[i] + ${dn(`${up('dout[i]')} * 2.0 * max(${up('inp[i]')}, 0.0)`)};
}`;

export const rmsnormBwdDinp = (N: number, C: number) => `${EN()}
@group(0) @binding(0) var<storage, read_write> dout: ${A1()};
@group(0) @binding(1) var<storage, read_write> inp: ${A1()};
@group(0) @binding(2) var<storage, read_write> w: ${A1()};
@group(0) @binding(3) var<storage, read_write> rrmsB: array<f32>;
@group(0) @binding(4) var<storage, read_write> dinp: ${A1()};
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let base = wid.x * ${C}u;
  let rrms = rrmsB[wid.x];
  var s = 0.0;
  for (var c = lid.x; c < ${C}u; c = c + 128u) {
    s = s + ${up('dout[base + c]')} * ${up('w[c]')} * ${up('inp[base + c]')} * rrms;
  }
  red[lid.x] = s;
  workgroupBarrier();
  for (var off = 64u; off > 0u; off = off >> 1u) {
    if (lid.x < off) { red[lid.x] = red[lid.x] + red[lid.x + off]; }
    workgroupBarrier();
  }
  let m = red[0] / ${C}.0;
  for (var c = lid.x; c < ${C}u; c = c + 128u) {
    let dnorm = ${up('dout[base + c]')} * ${up('w[c]')};
    let xhat = ${up('inp[base + c]')} * rrms;
    dinp[base + c] = dinp[base + c] + ${dn('(dnorm - xhat * m) * rrms')};
  }
}`;

// One workgroup per channel, reducing over the batch.
export const rmsnormBwdDw = (N: number, C: number) => `${EN()}
@group(0) @binding(0) var<storage, read_write> dout: ${A1()};
@group(0) @binding(1) var<storage, read_write> inp: ${A1()};
@group(0) @binding(2) var<storage, read_write> rrmsB: array<f32>;
@group(0) @binding(3) var<storage, read_write> dw: ${A1()};
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let c = wid.x;
  var s = 0.0;
  for (var r = lid.x; r < ${N}u; r = r + 256u) {
    s = s + ${up(`dout[r * ${C}u + c]`)} * ${up(`inp[r * ${C}u + c]`)} * rrmsB[r];
  }
  red[lid.x] = s;
  workgroupBarrier();
  for (var off = 128u; off > 0u; off = off >> 1u) {
    if (lid.x < off) { red[lid.x] = red[lid.x] + red[lid.x + off]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { dw[c] = dw[c] + ${dn('red[0]')}; }
}`;

export const crossentFwd = (BT: number, V: number) => `${EN()}
@group(0) @binding(0) var<storage, read_write> logits: ${A1()};
@group(0) @binding(1) var<storage, read_write> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> losses: array<f32>;
@group(0) @binding(3) var<storage, read_write> rowMax: array<f32>;
@group(0) @binding(4) var<storage, read_write> rowSum: array<f32>;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let base = wid.x * ${V}u;
  var mx = -1e30;
  for (var v = lid.x; v < ${V}u; v = v + 128u) { mx = max(mx, ${up('logits[base + v]')}); }
  red[lid.x] = mx;
  workgroupBarrier();
  for (var off = 64u; off > 0u; off = off >> 1u) {
    if (lid.x < off) { red[lid.x] = max(red[lid.x], red[lid.x + off]); }
    workgroupBarrier();
  }
  let m = red[0];
  workgroupBarrier();
  var s = 0.0;
  for (var v = lid.x; v < ${V}u; v = v + 128u) { s = s + exp(${up('logits[base + v]')} - m); }
  red[lid.x] = s;
  workgroupBarrier();
  for (var off = 64u; off > 0u; off = off >> 1u) {
    if (lid.x < off) { red[lid.x] = red[lid.x] + red[lid.x + off]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    rowMax[wid.x] = m;
    rowSum[wid.x] = red[0];
    losses[wid.x] = -(${up('logits[base + targets[wid.x]]')} - m - log(red[0]));
  }
}`;

export const crossentBwd = (BT: number, V: number) => {
  if (V % 4) throw new Error(`crossentBwd needs V % 4 == 0, got ${V}`);
  const V4 = V / 4;
  return `${EN()}
@group(0) @binding(0) var<storage, read_write> logits: ${A4()};
@group(0) @binding(1) var<storage, read_write> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> rowMax: array<f32>;
@group(0) @binding(3) var<storage, read_write> rowSum: array<f32>;
@group(0) @binding(4) var<storage, read_write> dlogits: ${A4()};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${BT * V4}u) { return; }
  let r = i / ${V4}u;
  let vb = (i % ${V4}u) * 4u;
  let p = exp(${up4('logits[i]')} - vec4<f32>(rowMax[r])) / rowSum[r];
  var ind = vec4<f32>();
  let tgt = targets[r];
  if (tgt >= vb && tgt < vb + 4u) { ind[tgt - vb] = 1.0; }
  dlogits[i] = dlogits[i] + ${dn4(`(p - ind) * ${1.0 / BT}`)};
}`;
};

export const encoderBwdWte = (B: number, T: number, C: number, V: number) => {
  const C4 = C / 4;
  return `${EN()}
@group(0) @binding(0) var<storage, read_write> ix: array<u32>;
@group(0) @binding(1) var<storage, read_write> dout: ${A4()};
@group(0) @binding(2) var<storage, read_write> dwte: ${A4()};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${V * C4}u) { return; }
  let v = i / ${C4}u;
  let c4 = i % ${C4}u;
  var acc = vec4<f32>();
  for (var bt = 0u; bt < ${B * T}u; bt = bt + 1u) {
    if (ix[bt] == v) { acc = acc + ${up4(`dout[bt * ${C4}u + c4]`)}; }
  }
  dwte[i] = dwte[i] + ${dn4('acc')};
}`;
};

// One thread per chunk, f32 accumulation.
export const gradNormPartial = (N: number, chunk: number, offset: number) => `${EN()}
@group(0) @binding(0) var<storage, read_write> g: ${A1()};
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${Math.ceil(N / chunk)}u) { return; }
  var acc = 0.0;
  let base = i * ${chunk}u;
  for (var j = 0u; j < ${chunk}u; j = j + 1u) {
    let k = base + j;
    if (k < ${N}u) {
      let gv = ${up('g[k]')};
      acc = acc + gv * gv;
    }
  }
  partials[${offset}u + i] = acc;
}`;

// Single-workgroup mean; keeps the
// eval readback to 4 bytes (large mapped readbacks proved unreliable in
// Chrome).
export const meanLosses = (BT: number) => `
@group(0) @binding(0) var<storage, read_write> losses: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var s = 0.0;
  for (var i = lid.x; i < ${BT}u; i = i + 256u) { s = s + losses[i]; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var off = 128u; off > 0u; off = off >> 1u) {
    if (lid.x < off) { red[lid.x] = red[lid.x] + red[lid.x + off]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { out[0] = red[0] / ${BT}.0; }
}`;

// Computes the clip scale; 0 on a NaN/inf norm, which skips the step.
export const gradNormFinalize = (nPartials: number, clip: number) => `
@group(0) @binding(0) var<storage, read_write> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> scale: array<f32>;
@compute @workgroup_size(1)
fn main() {
  var sum = 0.0;
  for (var i = 0u; i < ${nPartials}u; i = i + 1u) { sum = sum + partials[i]; }
  let norm = sqrt(sum);
  var s = ${clip} / max(norm, ${clip});
  if (norm != norm || norm > 1e30) { s = 0.0; }
  scale[0] = s;
}`;

// Updates f32 master weights and writes the f16 compute shadow in one pass.
// 2D grid: a 1D dispatch caps at 65,535 workgroups (16.7M threads), which
// the parameter pool exceeds.
export const adamw = (N: number, wd: number) => `${EN()}
struct U { t: f32, lr: f32, p0: f32, p1: f32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> p: array<f32>;
@group(0) @binding(2) var<storage, read_write> g: ${A1()};
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> scale: array<f32>;
@group(0) @binding(6) var<storage, read_write> ph: ${A1()};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * ${4096 * 256}u + gid.x;
  if (i >= ${N}u) { return; }
  let grad = ${up('g[i]')} * scale[0];
  let m_ = 0.9 * m[i] + 0.1 * grad;
  let v_ = 0.99 * v[i] + 0.01 * grad * grad;
  m[i] = m_;
  v[i] = v_;
  let mh = m_ / (1.0 - pow(0.9, u.t));
  let vh = v_ / (1.0 - pow(0.99, u.t));
  let pn = p[i] - u.lr * (mh / (sqrt(vh) + 1e-8) + ${wd} * p[i]);
  p[i] = pn;
  ph[i] = ${dn('pn')};
}`;

// Refreshes the f16 shadow from the f32 master after init or state import.
export const castHalf = (N: number) => `${EN()}
@group(0) @binding(0) var<storage, read_write> p: array<f32>;
@group(0) @binding(1) var<storage, read_write> ph: ${A1()};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * ${4096 * 256}u + gid.x;
  if (i >= ${N}u) { return; }
  ph[i] = ${dn('p[i]')};
}`;
