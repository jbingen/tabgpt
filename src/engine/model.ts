import * as K from './kernels';

function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

export interface GPTConfig {
  B: number;   // batch size
  T: number;   // sequence length (= max context)
  C: number;   // model width
  L: number;   // layers
  NH: number;  // heads
  V: number;   // vocab size
  lr: number;
  warmup: number;
  f16: boolean; // half-precision storage (requires the shader-f16 feature)
}

interface Op {
  pipeline: GPUComputePipeline;
  bind: GPUBindGroup;
  wgx: number;
  wgy: number;
}

// a binding is either a whole buffer or an aligned slice of a pool
type Bind = GPUBuffer | { buffer: GPUBuffer; offset: number; size: number };

// parameters, grads, Adam moments and activation grads live in single pooled
// buffers, sliced per tensor at 256-byte-aligned offsets. This turns ~100
// per-step clearBuffer calls into 2 and the optimizer into 3 dispatches.
const ALIGN = 128; // elements; 256 bytes even at f16, the offset alignment floor

interface PRec { name: string; shape: number[]; numel: number; off: number }
interface ARec { buf: GPUBuffer; numel: number; gOff: number }

const ceil = (a: number, b: number) => Math.ceil(a / b);
const alignUp = (n: number) => Math.ceil(n / ALIGN) * ALIGN;

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// modern-GPT architecture, modded-nanogpt style: RMSNorm, RoPE, no biases,
// ReLU^2 MLP, zero-init projections, embedding tied to the lm head.
export class GPT {
  readonly cfg: GPTConfig;
  numParams = 0;
  step = 0;
  paramRecords: PRec[] = [];

  private device: GPUDevice;
  private pipeCache = new Map<string, GPUComputePipeline>();
  private fwdOps: Op[] = [];
  private lossBwdOps: Op[] = [];
  private lossOp!: Op;
  private meanOp!: Op;
  private meanLoss!: GPUBuffer;
  private optOps: Op[] = [];

  private paramPool!: GPUBuffer;
  private paramHalf!: GPUBuffer;
  private gradPool!: GPUBuffer;
  private mPool!: GPUBuffer;
  private vPool!: GPUBuffer;
  private actGradPool!: GPUBuffer;
  private pTotal = 0;

  private ix: GPUBuffer;
  private targets: GPUBuffer;
  private uni: GPUBuffer;
  private losses: GPUBuffer;
  private logits: ARec;
  private stagingLoss: GPUBuffer;
  private stagingLogits: GPUBuffer;
  private castOp!: Op;
  private hb: number;

  constructor(device: GPUDevice, cfg: GPTConfig) {
    this.device = device;
    this.cfg = cfg;
    const { B, T, C, L, NH, V } = cfg;
    const BT = B * T;

    const rng = mulberry32(1337);
    const gauss = () => {
      // Box-Muller
      const u1 = Math.max(rng(), 1e-12);
      const u2 = rng();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    };

    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const mkBuf = (numel: number) => device.createBuffer({ size: numel * 4, usage: STORAGE });
    // "half" buffers: f16 when enabled, plain f32 otherwise
    const hb = cfg.f16 ? 2 : 4;
    this.hb = hb;
    const mkHalf = (numel: number) => device.createBuffer({ size: numel * hb, usage: STORAGE });

    this.ix = device.createBuffer({ size: BT * 4, usage: STORAGE });
    this.targets = device.createBuffer({ size: BT * 4, usage: STORAGE });
    this.uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.losses = mkBuf(BT);
    this.stagingLoss = device.createBuffer({ size: BT * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.stagingLogits = device.createBuffer({ size: V * hb, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    // parameter declaration is bookkeeping only; pools are allocated once
    // the padded total is known
    const inits: { off: number; data: Float32Array<ArrayBuffer> }[] = [];
    let pTotal = 0;
    const param = (name: string, shape: number[], init: 'gauss' | 'ones' | 'zeros', std: number): PRec => {
      const numel = shape.reduce((a, b) => a * b, 1);
      const data = new Float32Array(numel);
      if (init === 'gauss') for (let i = 0; i < numel; i++) data[i] = gauss() * std;
      else if (init === 'ones') data.fill(1);
      const rec: PRec = { name, shape, numel, off: pTotal };
      pTotal += alignUp(numel);
      inits.push({ off: rec.off, data });
      this.paramRecords.push(rec);
      this.numParams += numel;
      return rec;
    };

    const wte = param('wte', [V, C], 'gauss', 0.02);
    const layers = Array.from({ length: L }, (_, l) => ({
      ln1w: param(`h.${l}.ln1.weight`, [C], 'ones', 0),
      wqkv: param(`h.${l}.attn.qkv.weight`, [C, 3 * C], 'gauss', 0.02),
      wproj: param(`h.${l}.attn.proj.weight`, [C, C], 'zeros', 0),
      ln2w: param(`h.${l}.ln2.weight`, [C], 'ones', 0),
      wfc: param(`h.${l}.mlp.fc.weight`, [C, 4 * C], 'gauss', 0.02),
      wfcproj: param(`h.${l}.mlp.proj.weight`, [4 * C, C], 'zeros', 0),
    }));
    const lnfw = param('lnf.weight', [C], 'ones', 0);

    this.pTotal = pTotal;
    this.paramPool = mkBuf(pTotal);
    this.paramHalf = mkHalf(pTotal);
    this.gradPool = mkHalf(pTotal);
    this.mPool = mkBuf(pTotal);
    this.vPool = mkBuf(pTotal);
    for (const { off, data } of inits) device.queue.writeBuffer(this.paramPool, off * 4, data);

    // compute kernels read the half shadow; adamw/export use the f32 master
    const P = (r: PRec): Bind => ({ buffer: this.paramHalf, offset: r.off * hb, size: r.numel * hb });
    const G = (r: PRec): Bind => ({ buffer: this.gradPool, offset: r.off * hb, size: r.numel * hb });

    const grid2d = (threads: number): [number, number] => [
      Math.min(ceil(threads, 256), 4096),
      Math.ceil(threads / (4096 * 256)),
    ];
    const [cx, cy] = grid2d(pTotal);
    this.castOp = this.op(K.castHalf(pTotal), [this.paramPool, this.paramHalf], cx, cy);
    {
      const enc = device.createCommandEncoder();
      this.runOps(enc, [this.castOp]);
      device.queue.submit([enc.finish()]);
    }

    const CHUNK = 1024;
    const nPart = Math.ceil(pTotal / CHUNK);
    const partials = mkBuf(nPart);
    const gscale = mkBuf(1);
    this.optOps.push(
      this.op(K.gradNormPartial(pTotal, CHUNK, 0), [this.gradPool, partials], ceil(nPart, 256)),
      this.op(K.gradNormFinalize(nPart, 1.0), [partials, gscale], 1),
      this.op(K.adamw(pTotal, 0.01), [this.uni, this.paramPool, this.gradPool, this.mPool, this.vPool, gscale, this.paramHalf], ...grid2d(pTotal)),
    );

    let aTotal = 0;
    const act = (numel: number): ARec => {
      const rec: ARec = { buf: mkHalf(numel), numel, gOff: aTotal };
      aTotal += alignUp(numel);
      return rec;
    };

    const x0 = act(BT * C);
    const acts = Array.from({ length: L }, () => ({
      ln1: act(BT * C), ln1r: mkBuf(BT),
      qkv: act(BT * 3 * C), qkr: mkBuf(B * T * 2 * NH),
      preatt: act(B * NH * T * T), att: act(B * NH * T * T),
      atty: act(BT * C), attproj: act(BT * C), res2: act(BT * C),
      ln2: act(BT * C), ln2r: mkBuf(BT),
      fc: act(BT * 4 * C), relu: act(BT * 4 * C), fcproj: act(BT * C), res3: act(BT * C),
    }));
    const lnf = act(BT * C);
    const lnfr = mkBuf(BT);
    this.logits = act(BT * V);
    const rowMax = mkBuf(BT);
    const rowSum = mkBuf(BT);

    this.actGradPool = mkHalf(aTotal);
    const AG = (r: ARec): Bind => ({ buffer: this.actGradPool, offset: r.gOff * hb, size: r.numel * hb });

    this.fwdOps.push(this.op(K.embedFwd(B, T, C), [this.ix, P(wte), x0.buf], ceil(BT * C, 256)));

    let x = x0;
    for (let l = 0; l < L; l++) {
      const p = layers[l];
      const a = acts[l];
      this.fwdOps.push(
        this.op(K.rmsnormFwd(BT, C), [x.buf, P(p.ln1w), a.ln1.buf, a.ln1r], BT),
        this.op(K.matmulFwd(BT, C, 3 * C, false), [a.ln1.buf, P(p.wqkv), a.qkv.buf], ceil(3 * C, 64), ceil(BT, 64)),
        this.op(K.rope(B, T, C, NH, false), [a.qkv.buf], ceil(BT * C, 256)),
        this.op(K.qknormFwd(B, T, C, NH), [a.qkv.buf, a.qkr], ceil(B * T * 2 * NH, 256)),
        this.op(K.attnScores(B, T, C, NH), [a.qkv.buf, a.preatt.buf], ceil(B * NH * T * T, 256)),
        this.op(K.attnSoftmax(B, T, NH), [a.preatt.buf, a.att.buf], B * NH * T),
        this.op(K.attnAgg(B, T, C, NH), [a.att.buf, a.qkv.buf, a.atty.buf], ceil((BT * C) / 4, 256)),
        this.op(K.matmulFwd(BT, C, C, false), [a.atty.buf, P(p.wproj), a.attproj.buf], ceil(C, 64), ceil(BT, 64)),
        this.op(K.residualFwd(BT * C), [x.buf, a.attproj.buf, a.res2.buf], ceil(BT * C, 256)),
        this.op(K.rmsnormFwd(BT, C), [a.res2.buf, P(p.ln2w), a.ln2.buf, a.ln2r], BT),
        this.op(K.matmulFwd(BT, C, 4 * C, false), [a.ln2.buf, P(p.wfc), a.fc.buf], ceil(4 * C, 64), ceil(BT, 64)),
        this.op(K.relu2Fwd(BT * 4 * C), [a.fc.buf, a.relu.buf], ceil(BT * 4 * C, 256)),
        this.op(K.matmulFwd(BT, 4 * C, C, false), [a.relu.buf, P(p.wfcproj), a.fcproj.buf], ceil(C, 64), ceil(BT, 64)),
        this.op(K.residualFwd(BT * C), [a.res2.buf, a.fcproj.buf, a.res3.buf], ceil(BT * C, 256)),
      );
      x = a.res3;
    }

    this.fwdOps.push(
      this.op(K.rmsnormFwd(BT, C), [x.buf, P(lnfw), lnf.buf, lnfr], BT),
      // tied head: logits = lnf @ wte^T
      this.op(K.matmulABt(BT, V, C, false), [lnf.buf, P(wte), this.logits.buf], ceil(V, 64), ceil(BT, 64)),
    );

    this.lossOp = this.op(K.crossentFwd(BT, V), [this.logits.buf, this.targets, this.losses, rowMax, rowSum], BT);
    this.meanLoss = mkBuf(1);
    this.meanOp = this.op(K.meanLosses(BT), [this.losses, this.meanLoss], 1);
    this.lossBwdOps.push(
      this.lossOp,
      this.op(K.crossentBwd(BT, V), [this.logits.buf, this.targets, rowMax, rowSum, AG(this.logits)], ceil((BT * V) / 4, 256)),
      // dlnf = dlogits @ wte   (wte is [V,C] row-major, exactly matmulFwd's w layout)
      this.op(K.matmulFwd(BT, V, C, false), [AG(this.logits), P(wte), AG(lnf)], ceil(C, 64), ceil(BT, 64)),
      // dwte(head) += dlogits^T @ lnf
      this.op(K.matmulBwdDw(BT, V, C), [AG(this.logits), lnf.buf, G(wte)], ceil(C, 64), ceil(V, 64)),
      this.op(K.rmsnormBwdDinp(BT, C), [AG(lnf), x.buf, P(lnfw), lnfr, AG(x)], BT),
      this.op(K.rmsnormBwdDw(BT, C), [AG(lnf), x.buf, lnfr, G(lnfw)], C),
    );

    for (let l = L - 1; l >= 0; l--) {
      const p = layers[l];
      const a = acts[l];
      const xin = l === 0 ? x0 : acts[l - 1].res3;
      this.lossBwdOps.push(
        // res3 = res2 + fcproj
        this.op(K.residualBwd(BT * C), [AG(a.res3), AG(a.res2), AG(a.fcproj)], ceil(BT * C, 256)),
        // fcproj = relu @ wfcproj
        this.op(K.matmulBwdDinp(BT, 4 * C, C), [AG(a.fcproj), P(p.wfcproj), AG(a.relu)], ceil(4 * C, 64), ceil(BT, 64)),
        this.op(K.matmulBwdDw(BT, 4 * C, C), [a.relu.buf, AG(a.fcproj), G(p.wfcproj)], ceil(C, 64), ceil(4 * C, 64)),
        // relu^2
        this.op(K.relu2Bwd(BT * 4 * C), [AG(a.relu), a.fc.buf, AG(a.fc)], ceil(BT * 4 * C, 256)),
        // fc = ln2 @ wfc
        this.op(K.matmulBwdDinp(BT, C, 4 * C), [AG(a.fc), P(p.wfc), AG(a.ln2)], ceil(C, 64), ceil(BT, 64)),
        this.op(K.matmulBwdDw(BT, C, 4 * C), [a.ln2.buf, AG(a.fc), G(p.wfc)], ceil(4 * C, 64), ceil(C, 64)),
        // ln2(res2)
        this.op(K.rmsnormBwdDinp(BT, C), [AG(a.ln2), a.res2.buf, P(p.ln2w), a.ln2r, AG(a.res2)], BT),
        this.op(K.rmsnormBwdDw(BT, C), [AG(a.ln2), a.res2.buf, a.ln2r, G(p.ln2w)], C),
        // res2 = x + attproj
        this.op(K.residualBwd(BT * C), [AG(a.res2), AG(xin), AG(a.attproj)], ceil(BT * C, 256)),
        // attproj = atty @ wproj
        this.op(K.matmulBwdDinp(BT, C, C), [AG(a.attproj), P(p.wproj), AG(a.atty)], ceil(C, 64), ceil(BT, 64)),
        this.op(K.matmulBwdDw(BT, C, C), [a.atty.buf, AG(a.attproj), G(p.wproj)], ceil(C, 64), ceil(C, 64)),
        // attention
        this.op(K.attnAggBwdDatt(B, T, C, NH), [AG(a.atty), a.qkv.buf, AG(a.att)], ceil(B * NH * T * T, 256)),
        this.op(K.attnAggBwdDv(B, T, C, NH), [AG(a.atty), a.att.buf, AG(a.qkv)], ceil((BT * C) / 4, 256)),
        this.op(K.attnSoftmaxBwd(B, T, NH), [AG(a.att), a.att.buf, AG(a.preatt)], B * NH * T),
        this.op(K.attnScoresBwdDq(B, T, C, NH), [AG(a.preatt), a.qkv.buf, AG(a.qkv)], ceil((BT * C) / 4, 256)),
        this.op(K.attnScoresBwdDk(B, T, C, NH), [AG(a.preatt), a.qkv.buf, AG(a.qkv)], ceil((BT * C) / 4, 256)),
        this.op(K.qknormBwd(B, T, C, NH), [a.qkv.buf, a.qkr, AG(a.qkv)], ceil(B * T * 2 * NH, 256)),
        // undo the rotation on dq/dk (RoPE backward is the inverse rotation)
        this.op(K.rope(B, T, C, NH, true), [AG(a.qkv)], ceil(BT * C, 256)),
        // qkv = ln1 @ wqkv
        this.op(K.matmulBwdDinp(BT, C, 3 * C), [AG(a.qkv), P(p.wqkv), AG(a.ln1)], ceil(C, 64), ceil(BT, 64)),
        this.op(K.matmulBwdDw(BT, C, 3 * C), [a.ln1.buf, AG(a.qkv), G(p.wqkv)], ceil(3 * C, 64), ceil(C, 64)),
        // ln1(x)
        this.op(K.rmsnormBwdDinp(BT, C), [AG(a.ln1), xin.buf, P(p.ln1w), a.ln1r, AG(xin)], BT),
        this.op(K.rmsnormBwdDw(BT, C), [AG(a.ln1), xin.buf, a.ln1r, G(p.ln1w)], C),
      );
    }
    this.lossBwdOps.push(
      this.op(K.encoderBwdWte(B, T, C, V), [this.ix, AG(x0), G(wte)], ceil((V * C) / 4, 256)),
    );
  }

  private op(code: string, buffers: Bind[], wgx: number, wgy = 1): Op {
    let pipeline = this.pipeCache.get(code);
    if (!pipeline) {
      const module = this.device.createShaderModule({ code });
      pipeline = this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      this.pipeCache.set(code, pipeline);
    }
    const bind = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((b, binding) => ({
        binding,
        resource: b instanceof GPUBuffer ? { buffer: b } : b,
      })),
    });
    return { pipeline, bind, wgx, wgy };
  }

  /** Submits ops in short command buffers so the window compositor can
      interleave its own GPU work between them. A single full-step command
      buffer (~100ms) visibly stalls the UI on macOS. */
  private submitChunked(ops: Op[], chunk = 16, pre?: (enc: GPUCommandEncoder) => void) {
    for (let i = 0; i < ops.length; i += chunk) {
      const enc = this.device.createCommandEncoder();
      if (i === 0 && pre) pre(enc);
      this.runOps(enc, ops.slice(i, i + chunk));
      this.device.queue.submit([enc.finish()]);
    }
  }

  private runOps(enc: GPUCommandEncoder, ops: Op[]) {
    const pass = enc.beginComputePass();
    for (const o of ops) {
      pass.setPipeline(o.pipeline);
      pass.setBindGroup(0, o.bind);
      pass.dispatchWorkgroups(o.wgx, o.wgy);
    }
    pass.end();
  }

  /** One training step on the given batch. Returns mean loss, or NaN when
      readLoss is false (skipping the readback keeps the GPU pipeline full). */
  async trainStep(x: Uint32Array<ArrayBuffer>, y: Uint32Array<ArrayBuffer>, readLoss = true): Promise<number> {
    const { B, T, lr, warmup } = this.cfg;
    this.step++;
    // linear warmup, cosine decay to 5%, then hold. Decay is deliberately
    // fast: at this width, a slowly-decaying lr produced sustained loss
    // regression past ~6k steps (noise-floor effect).
    const decaySteps = 8000;
    let curLr: number;
    if (this.step < warmup) {
      curLr = (lr * this.step) / warmup;
    } else if (this.step < warmup + decaySteps) {
      const p = (this.step - warmup) / decaySteps;
      curLr = lr * (0.05 + 0.475 * (1 + Math.cos(Math.PI * p)));
    } else {
      curLr = lr * 0.05;
    }
    this.device.queue.writeBuffer(this.uni, 0, new Float32Array([this.step, curLr, 0, 0]));
    this.device.queue.writeBuffer(this.ix, 0, x);
    this.device.queue.writeBuffer(this.targets, 0, y);

    this.submitChunked(this.fwdOps, 16, (enc) => {
      enc.clearBuffer(this.gradPool);
      enc.clearBuffer(this.actGradPool);
    });
    this.submitChunked(this.lossBwdOps, 16);
    const enc = this.device.createCommandEncoder();
    this.runOps(enc, this.optOps);
    if (readLoss) enc.copyBufferToBuffer(this.losses, 0, this.stagingLoss, 0, B * T * 4);
    this.device.queue.submit([enc.finish()]);
    if (!readLoss) return NaN;

    await this.stagingLoss.mapAsync(GPUMapMode.READ);
    const arr = new Float32Array(this.stagingLoss.getMappedRange());
    let sum = 0;
    for (let i = 0; i < arr.length; i++) sum += arr[i];
    const loss = sum / arr.length;
    this.stagingLoss.unmap();
    return loss;
  }

  /** Diagnostic for a non-finite eval: reads back the loss buffer and the
      first bad row's logits to localize the source. */
  async debugVal(): Promise<string> {
    const { B, T, V } = this.cfg;
    const BT = B * T;
    const readBuf = async (src: GPUBuffer, offBytes: number, bytes: number): Promise<ArrayBuffer> => {
      const st = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(src, offBytes, st, 0, bytes);
      this.device.queue.submit([enc.finish()]);
      await st.mapAsync(GPUMapMode.READ);
      const ab = st.getMappedRange().slice(0);
      st.unmap();
      st.destroy();
      return ab;
    };
    const losses = new Float32Array(await readBuf(this.losses, 0, BT * 4));
    let r = -1;
    let bad = 0;
    for (let i = 0; i < BT; i++) if (!isFinite(losses[i])) { if (r < 0) r = i; bad++; }
    if (r < 0) {
      let mn = Infinity, mx = -Infinity, sum = 0;
      for (let i = 0; i < BT; i++) { mn = Math.min(mn, losses[i]); mx = Math.max(mx, losses[i]); sum += losses[i]; }
      return `losses all finite: mean=${(sum / BT).toFixed(3)} min=${mn.toFixed(3)} max=${mx.toFixed(3)}`;
    }
    const raw = await readBuf(this.logits.buf, r * V * this.hb, V * this.hb);
    let vals: Float32Array;
    if (this.cfg.f16) {
      const u = new Uint16Array(raw);
      vals = new Float32Array(u.length);
      for (let v = 0; v < u.length; v++) vals[v] = halfToFloat(u[v]);
    } else {
      vals = new Float32Array(raw);
    }
    let nInf = 0, nNan = 0, mx = -Infinity, mn = Infinity;
    for (const x of vals) {
      if (Number.isNaN(x)) nNan++;
      else if (!isFinite(x)) nInf++;
      else { mx = Math.max(mx, x); mn = Math.min(mn, x); }
    }
    const tgt = new Uint32Array(await readBuf(this.targets, r * 4, 4))[0];
    return `bad rows ${bad}/${BT}, first row ${r}: loss=${losses[r]}, logits inf=${nInf} nan=${nNan} `
      + `min=${mn.toFixed(1)} max=${mx.toFixed(1)}, target=${tgt} logit[target]=${vals[tgt]}`;
  }

  /** Forward + loss only, no gradients or optimizer. For held-out
      evaluation. The mean is reduced on-GPU and read back through a 4-byte
      staging buffer; large shared staging readbacks were observed to return
      corrupted data in Chrome. */
  async valStep(x: Uint32Array<ArrayBuffer>, y: Uint32Array<ArrayBuffer>): Promise<number> {
    const { B, T } = this.cfg;
    this.device.queue.writeBuffer(this.ix, 0, x);
    this.device.queue.writeBuffer(this.targets, 0, y);
    const enc = this.device.createCommandEncoder();
    this.runOps(enc, this.fwdOps);
    this.runOps(enc, [this.lossOp]);
    this.runOps(enc, [this.meanOp]);
    this.device.queue.submit([enc.finish()]);
    // 4-byte readback with one retry; see the note on the method
    for (let attempt = 0; attempt < 2; attempt++) {
      const staging = this.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const enc2 = this.device.createCommandEncoder();
      enc2.copyBufferToBuffer(this.meanLoss, 0, staging, 0, 4);
      this.device.queue.submit([enc2.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const v = new Float32Array(staging.getMappedRange())[0];
      staging.unmap();
      staging.destroy();
      if (isFinite(v) || attempt === 1) return v;
    }
    return NaN;
  }

  /** Autoregressive sampling. Runs the forward pass with the context in batch row 0.
      topV restricts sampling to the real vocabulary when V is padded.
      Returns each sampled token with the model's probability for it. */
  async generate(prefix: number[], n: number, temperature = 0.8, topV?: number): Promise<{ tokens: number[]; probs: number[] }> {
    const { B, T, V } = this.cfg;
    const tv = topV ?? V;
    const ctx = prefix.slice();
    const out: number[] = [];
    const outProbs: number[] = [];
    const arr = new Uint32Array(B * T);
    for (let i = 0; i < n; i++) {
      arr.fill(0);
      const win = ctx.slice(Math.max(0, ctx.length - T));
      for (let j = 0; j < win.length; j++) arr[j] = win[j];
      this.device.queue.writeBuffer(this.ix, 0, arr);

      const enc = this.device.createCommandEncoder();
      this.runOps(enc, this.fwdOps);
      enc.copyBufferToBuffer(this.logits.buf, (win.length - 1) * V * this.hb, this.stagingLogits, 0, V * this.hb);
      this.device.queue.submit([enc.finish()]);

      await this.stagingLogits.mapAsync(GPUMapMode.READ);
      let logits: Float32Array;
      if (this.cfg.f16) {
        const raw = new Uint16Array(this.stagingLogits.getMappedRange()).slice();
        this.stagingLogits.unmap();
        logits = new Float32Array(raw.length);
        for (let v = 0; v < raw.length; v++) logits[v] = halfToFloat(raw[v]);
      } else {
        logits = new Float32Array(this.stagingLogits.getMappedRange()).slice();
        this.stagingLogits.unmap();
      }

      // top-k sampling: restrict the draw to the k highest logits
      const K = 40;
      const order = Array.from({ length: tv }, (_, v) => v)
        .sort((a2, b2) => logits[b2] - logits[a2])
        .slice(0, K);
      const mx = logits[order[0]];
      let sum = 0;
      const probs = new Float64Array(K);
      for (let k = 0; k < K; k++) { probs[k] = Math.exp((logits[order[k]] - mx) / temperature); sum += probs[k]; }
      let r = Math.random() * sum;
      let tok = order[K - 1];
      let chosenP = probs[K - 1];
      for (let k = 0; k < K; k++) { r -= probs[k]; if (r <= 0) { tok = order[k]; chosenP = probs[k]; break; } }
      ctx.push(tok);
      out.push(tok);
      outProbs.push(chosenP / sum);
    }
    return { tokens: out, probs: outProbs };
  }

  /** Full training state (weights + Adam moments), for resuming across reloads.
      Raw padded pool layout; format is tied to fmt in the saved blob. */
  async exportState(): Promise<{ step: number; data: Float32Array }> {
    const n = this.pTotal;
    const staging = this.device.createBuffer({
      size: n * 3 * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.paramPool, 0, staging, 0, n * 4);
    enc.copyBufferToBuffer(this.mPool, 0, staging, n * 4, n * 4);
    enc.copyBufferToBuffer(this.vPool, 0, staging, n * 8, n * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();
    staging.destroy();
    return { step: this.step, data };
  }

  importState(data: Float32Array<ArrayBuffer>, step: number) {
    const n = this.pTotal;
    if (data.length !== n * 3) throw new Error('saved state does not match this model');
    this.device.queue.writeBuffer(this.paramPool, 0, data, 0, n);
    this.device.queue.writeBuffer(this.mPool, 0, data, n, n);
    this.device.queue.writeBuffer(this.vPool, 0, data, n * 2, n);
    const enc = this.device.createCommandEncoder();
    this.runOps(enc, [this.castOp]);
    this.device.queue.submit([enc.finish()]);
    this.step = step;
  }

  /** Read a single weight tensor back from GPU memory. */
  async readParam(name: string): Promise<Float32Array> {
    const r = this.paramRecords.find((p) => p.name === name);
    if (!r) throw new Error(`no param named ${name}`);
    const staging = this.device.createBuffer({
      size: r.numel * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.paramPool, r.off * 4, staging, 0, r.numel * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();
    staging.destroy();
    return out;
  }

  /** Load weights from a .safetensors export made by exportWeights. The file
      carries no optimizer state, so the Adam moments restart from zero; the
      step comes from the file's metadata so the LR schedule picks up where the
      exported run left off. */
  importWeights(buf: ArrayBuffer): number {
    const dv = new DataView(buf);
    if (buf.byteLength < 16) throw new Error('not a safetensors file');
    const hlen = Number(dv.getBigUint64(0, true));
    if (hlen <= 0 || 8 + hlen > buf.byteLength) throw new Error('not a safetensors file');
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hlen)));
    const body = 8 + hlen;
    for (const r of this.paramRecords) {
      const t = header[r.name];
      if (!t || t.dtype !== 'F32' || String(t.shape) !== String(r.shape)) {
        throw new Error(`checkpoint does not match this model (${r.name})`);
      }
      const [a, b] = t.data_offsets as [number, number];
      if (b - a !== r.numel * 4 || body + b > buf.byteLength) {
        throw new Error(`checkpoint does not match this model (${r.name})`);
      }
      const data = new Float32Array(buf.slice(body + a, body + b));
      for (let i = 0; i < data.length; i++) {
        if (!isFinite(data[i])) throw new Error('checkpoint contains non-finite values');
      }
      this.device.queue.writeBuffer(this.paramPool, r.off * 4, data);
    }
    const enc = this.device.createCommandEncoder();
    enc.clearBuffer(this.mPool);
    enc.clearBuffer(this.vPool);
    this.runOps(enc, [this.castOp]);
    this.device.queue.submit([enc.finish()]);
    const step = Number(header.__metadata__?.step ?? 0);
    this.step = Number.isFinite(step) && step > 0 ? Math.floor(step) : 0;
    return this.step;
  }

  /** Snapshot the live weights out of GPU memory as a .safetensors blob. */
  async exportWeights(meta: Record<string, string>): Promise<Blob> {
    const total = this.paramRecords.reduce((s, r) => s + r.numel, 0);
    const staging = this.device.createBuffer({
      size: total * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this.device.createCommandEncoder();
    const header: Record<string, unknown> = { __metadata__: meta };
    let off = 0;
    for (const r of this.paramRecords) {
      enc.copyBufferToBuffer(this.paramPool, r.off * 4, staging, off * 4, r.numel * 4);
      header[r.name] = { dtype: 'F32', shape: r.shape, data_offsets: [off * 4, (off + r.numel) * 4] };
      off += r.numel;
    }
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const data = new Uint8Array(staging.getMappedRange()).slice();
    staging.unmap();
    staging.destroy();

    const json = new TextEncoder().encode(JSON.stringify(header));
    // pad the header to 8 bytes with trailing spaces, per the safetensors spec
    const padded = new Uint8Array(Math.ceil(json.length / 8) * 8).fill(0x20);
    padded.set(json);
    const len = new ArrayBuffer(8);
    new DataView(len).setBigUint64(0, BigInt(padded.length), true);
    return new Blob([len, padded, data], { type: 'application/octet-stream' });
  }
}
