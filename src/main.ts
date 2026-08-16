import { GPT } from './engine/model';
import { setF16 } from './engine/kernels';
import { loadTokenData, sampleBatch, decodeTokens, encodeText, type TokenData } from './engine/data';

const $ = (id: string) => document.getElementById(id)!;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const EASE = 'cubic-bezier(0.23, 1, 0.32, 1)';

const baseCfg = { T: 128, C: 384, L: 10, NH: 8, lr: 6e-4, warmup: 300 };

// batch size adapts to the machine: bigger batches are faster but need more
// GPU memory. batchCap ratchets down if the first steps come out too slow.
let batchCap = 64;
function pickBatch(adapter: GPUAdapter): number {
  const mem = (navigator as any).deviceMemory ?? 8;
  const maxBuf = adapter.limits.maxBufferSize;
  let b = 4;
  if (maxBuf >= 600_000_000 && mem >= 4) b = 8;
  if (maxBuf >= 1_200_000_000 && mem >= 8) b = 16;
  return Math.min(b, batchCap);
}

let runToken = 0;
let device: GPUDevice | null = null;
let running = true;
let pendingPrompt: string | null = null;

const losses: number[] = [];
const valPts: { i: number; v: number }[] = [];

// the chart aggregates into permanent min/max buckets (RRD-style): once a
// bucket is written it never changes, so the drawn shape is stable and only
// the right edge moves. When buckets outgrow the chart, pairs merge for good.
let chMin: number[] = [];
let chMax: number[] = [];
let chSize = 1;
let chN = 0;
let chAmin = Infinity;
let chAmax = -Infinity;

function chartPush(l: number) {
  if (l < chAmin) chAmin = l;
  if (l > chAmax) chAmax = l;
  chN++;
  if (chN >= chSize) {
    chMin.push(chAmin);
    chMax.push(chAmax);
    chN = 0;
    chAmin = Infinity;
    chAmax = -Infinity;
    if (chMin.length >= 2048) {
      const nmin: number[] = [];
      const nmax: number[] = [];
      for (let i = 0; i + 1 < chMin.length; i += 2) {
        nmin.push(Math.min(chMin[i], chMin[i + 1]));
        nmax.push(Math.max(chMax[i], chMax[i + 1]));
      }
      chMin = nmin;
      chMax = nmax;
      chSize *= 2;
    }
  }
}

function chartReset() {
  chMin = [];
  chMax = [];
  chSize = 1;
  chN = 0;
  chAmin = Infinity;
  chAmax = -Infinity;
}

function chartRebuild(arr: number[]) {
  chartReset();
  for (const l of arr) chartPush(l);
}
let ema = 0;
let sessionTokens = 0;
let saveNow: (() => void) | null = null;
let badRuns = 0;

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('backprop', 1);
    req.onupgradeneeded = () => req.result.createObjectStore('state');
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

// the state blob is hundreds of MB at larger model sizes; stored in 64MB
// chunks because single giant IndexedDB values fail
async function idbPut(meta: Record<string, unknown>, data: ArrayBuffer) {
  const db = await openDb();
  await new Promise((resolve, reject) => {
    const tx = db.transaction('state', 'readwrite');
    const store = tx.objectStore('state');
    const CH = 64 << 20;
    const chunks = Math.ceil(data.byteLength / CH);
    store.put({ ...meta, chunks, bytes: data.byteLength }, 'v1');
    for (let i = 0; i < chunks; i++) store.put(data.slice(i * CH, (i + 1) * CH), `v1c${i}`);
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
  db.close();
}

async function idbGet(): Promise<any> {
  const db = await openDb();
  const get = (key: string) => new Promise<any>((resolve, reject) => {
    const req = db.transaction('state', 'readonly').objectStore('state').get(key);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  const meta = await get('v1');
  if (!meta) { db.close(); return null; }
  if (meta.chunks && !meta.data) {
    const buf = new Uint8Array(meta.bytes);
    let off = 0;
    for (let i = 0; i < meta.chunks; i++) {
      const c: ArrayBuffer | undefined = await get(`v1c${i}`);
      if (!c) { db.close(); return null; }
      buf.set(new Uint8Array(c), off);
      off += c.byteLength;
    }
    meta.data = buf.buffer;
  }
  db.close();
  return meta;
}

function setStatus(kind: 'ok' | 'paused' | 'error' | 'wait', text: string) {
  $('statusdot').dataset.kind = kind;
  $('statustext').textContent = text;
  // while starting up (or broken) the stats and chart are all placeholders,
  // so the footer status alone reads as a dead page. Mirror it up top.
  const top = $('status');
  top.dataset.kind = kind;
  top.textContent = kind === 'wait' || kind === 'error' ? text : '';
}

function drawChart() {
  const canvas = $('chart') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d')!;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w === 0) return;
  if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const m = chMin.length;
  if (m < 2) return;

  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < m; i++) {
    if (chMin[i] < lo) lo = chMin[i];
    if (chMax[i] > hi) hi = chMax[i];
  }
  for (const p of valPts) {
    if (p.v < lo) lo = p.v;
    if (p.v > hi) hi = p.v;
  }
  lo *= 0.98;
  hi *= 1.02;
  if (!isFinite(lo) || !isFinite(hi) || hi <= lo) return;
  const py = (l: number) => h - ((l - lo) / (hi - lo)) * h;

  ctx.strokeStyle = 'oklch(0.28 0.02 260 / 0.08)';
  ctx.fillStyle = 'oklch(0.52 0.02 255 / 0.6)';
  ctx.font = '500 10.5px -apple-system, system-ui, sans-serif';
  ctx.lineWidth = 1;
  for (let g = Math.ceil(lo); g <= hi && g < Math.ceil(lo) + 40; g++) {
    const y = py(g);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    ctx.fillText(String(g), 4, y - 4);
  }

  // stable envelope over the permanent buckets. Early in a run each bucket
  // holds a single reading and the band collapses to nothing, so the edges
  // are also stroked: a young run reads as a line, a long one as a band
  const bx = (i: number) => (i / (m - 1)) * w;
  ctx.fillStyle = 'oklch(0.6 0.2 255 / 0.85)';
  ctx.beginPath();
  ctx.moveTo(bx(0), py(chMax[0]));
  for (let i = 1; i < m; i++) ctx.lineTo(bx(i), py(chMax[i]));
  for (let i = m - 1; i >= 0; i--) ctx.lineTo(bx(i), Math.max(py(chMin[i]), py(chMax[i]) + 1));
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = 'oklch(0.6 0.2 255 / 0.9)';
  ctx.lineWidth = 1.25;
  ctx.lineJoin = 'round';
  for (const edge of [chMax, chMin]) {
    ctx.beginPath();
    ctx.moveTo(bx(0), py(edge[0]));
    for (let i = 1; i < m; i++) ctx.lineTo(bx(i), py(edge[i]));
    ctx.stroke();
  }

  if (valPts.length > 1) {
    ctx.strokeStyle = 'oklch(0.28 0.02 260 / 0.45)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const nl = Math.max(1, losses.length - 1);
    valPts.forEach((p, k) => {
      const X = (Math.min(p.i, nl) / nl) * w;
      const Y = py(p.v);
      if (k === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
    });
    ctx.stroke();
  }
}

async function doSample(model: GPT, ds: TokenData) {
  const { tokens } = await model.generate(encodeText(ds, '\n'), 80, 0.8, ds.V);
  const text = decodeTokens(ds, tokens);
  const live = $('livesample');
  live.textContent = text;
  live.animate([{ opacity: 0.35 }, { opacity: 1 }], { duration: 350, easing: EASE });
  const words = text.toLowerCase().split(/[^\p{L}']+/u).filter((w) => w.length >= 2);
  const real = words.filter((w) => ds.words.has(w));
  $('realwords').textContent = words.length ? `${Math.round((real.length / words.length) * 100)}%` : '—';
}

async function doPrompt(model: GPT, ds: TokenData, raw: string) {
  const prefix = encodeText(ds, raw);
  const seed = prefix.length ? prefix : encodeText(ds, '\n');
  const { tokens } = await model.generate(seed, 60, 0.8, ds.V);
  const el = $('promptout');
  el.textContent = '';
  const pre = document.createElement('span');
  pre.className = 'prompt-prefix';
  pre.textContent = decodeTokens(ds, seed);
  const rest = document.createElement('span');
  rest.textContent = decodeTokens(ds, tokens);
  el.append(pre, rest);
  el.animate([{ opacity: 0.35 }, { opacity: 1 }], { duration: 350, easing: EASE });
}

async function startTraining() {
  const my = ++runToken;
  running = true;
  $('toggle').textContent = 'pause';

  try {
    if (device) { device.destroy(); device = null; }
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
    setStatus('wait', 'downloading the text');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No GPU adapter found.');
    // the activation pool outgrows the default 256MB buffer cap at batch 16
    const hasF16 = adapter.features.has('shader-f16');
    setF16(hasF16);
    device = await adapter.requestDevice({
      requiredFeatures: hasF16 ? ['shader-f16' as GPUFeatureName] : [],
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    });
    device.lost.then((info) => {
      if (my === runToken) { setStatus('error', 'GPU device lost'); console.error('device lost:', info.message); }
    });
    device.onuncapturederror = (e) => console.error('webgpu error:', (e as GPUUncapturedErrorEvent).error.message);
    const ds = await loadTokenData('', (got, total) => {
      const mb = (n: number) => (n / 1e6).toFixed(0);
      setStatus('wait', total > 0
        ? `downloading the text, ${mb(got)} of ${mb(total)} MB`
        : `downloading the text, ${mb(got)} MB`);
    });
    if (my !== runToken) return;

    setStatus('wait', 'compiling shaders');
    await sleep(30);
    // vec4 matmul kernels need dims divisible by 4; pad tokens are never sampled
    const cfg = { ...baseCfg, B: pickBatch(adapter), V: Math.ceil(ds.V / 4) * 4, f16: hasF16 };
    const model = new GPT(device, cfg);
    const mb = (model.numParams * 4) / 1e6;
    $('params').textContent = `${(model.numParams / 1e6).toFixed(2)}M`;
    console.log(`tabgpt: ${model.numParams} params, vocab ${ds.V}, ${ds.tokens.length} tokens`);

    // resume a previous run if the saved state matches this config + corpus
    const cfgKey = JSON.stringify(cfg);
    // the corpus generation participates in the resume key, so a model
    // trained on an earlier corpus generation retires when it is replaced
    const vocabKey = ds.gen > 1 ? `bpe:${ds.V}:tok16k:g${ds.gen}` : `bpe:${ds.V}:tok16k`;
    try {
      const saved = await idbGet();
      if (saved && saved.fmt === 5 && saved.config === cfgKey && saved.vocab === vocabKey && my === runToken) {
        // a save poisoned by NaN/inf weights would resume straight back into
        // divergence; scan before trusting it
        const state = new Float32Array(saved.data);
        let bad = false;
        for (let i = 0; i < state.length; i++) {
          if (!isFinite(state[i])) { bad = true; break; }
        }
        if (bad) throw new Error('saved state contains non-finite values, starting fresh');
        model.importState(state, saved.step);
        ema = saved.ema;
        sessionTokens = saved.sessionTokens;
        losses.length = 0;
        for (const l of saved.losses) losses.push(l);
        chartRebuild(losses);
        valPts.length = 0;
        if (saved.valPts) for (const p of saved.valPts) { if (isFinite(p.v)) valPts.push(p); }
        if (valPts.length) $('valloss').textContent = valPts[valPts.length - 1].v.toFixed(3);
        console.log(`tabgpt: resumed from step ${saved.step}`);
      }
    } catch (e) {
      console.warn('tabgpt: could not resume saved state', e);
    }

    let saving = false;
    saveNow = () => {
      if (saving || model.step === 0 || my !== runToken) return;
      saving = true;
      model.exportState()
        .then(({ step, data }) => idbPut({
          fmt: 5, config: cfgKey, vocab: vocabKey, step, ema, sessionTokens,
          losses: Float32Array.from(losses), valPts: valPts.slice(),
        }, data.buffer as ArrayBuffer))
        .then(() => console.log(`tabgpt: state saved at step ${model.step}`))
        .catch((e) => console.warn('tabgpt: SAVE FAILED', e))
        .finally(() => { saving = false; });
    };

    setStatus('ok', 'training');

    let exporting = false;
    const download = async () => {
      if (exporting || model.step === 0 || my !== runToken) return;
      exporting = true;
      try {
        const blob = await model.exportWeights({
          format: 'tabgpt',
          step: String(model.step),
          train_loss_ema: ema.toFixed(4),
          dataset: 'HuggingFaceFW/fineweb-2:nob_Latn',
          config: cfgKey,
          vocab: vocabKey,
        });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `tabgpt-nob-step${model.step}.safetensors`;
        a.click();
        URL.revokeObjectURL(a.href);
      } finally {
        exporting = false;
      }
    };
    $('download').onclick = download;

    // continue from a previously downloaded .safetensors. Weights only: the
    // Adam moments restart, so expect a small loss bump for the first steps
    const ckptfile = $('ckptfile') as HTMLInputElement;
    $('upload').onclick = () => { if (my === runToken) ckptfile.click(); };
    ckptfile.onchange = async () => {
      const f = ckptfile.files?.[0];
      ckptfile.value = '';
      if (!f || my !== runToken) return;
      try {
        const step = model.importWeights(await f.arrayBuffer());
        ema = 0;
        losses.length = 0;
        chartReset();
        valPts.length = 0;
        $('valloss').textContent = '—';
        $('lossema').textContent = '—';
        $('step').textContent = String(step);
        nextSampleStep = 20;
        while (nextSampleStep <= step) {
          nextSampleStep = Math.min(Math.round(nextSampleStep * 1.7), nextSampleStep + 500);
        }
        saveNow?.();
        console.log(`tabgpt: loaded checkpoint ${f.name} at step ${step}`);
      } catch (e) {
        console.warn('tabgpt: checkpoint load failed', e);
        setStatus('error', e instanceof Error ? e.message : 'could not read that file');
        setTimeout(() => {
          if (my === runToken) setStatus(running ? 'ok' : 'paused', running ? 'training' : 'paused');
        }, 5000);
      }
    };

    let tokPerSec = 0;
    let accMs = 0;
    let accTok = 0;
    let nextSampleStep = 20;
    while (nextSampleStep <= model.step) {
      nextSampleStep = Math.min(Math.round(nextSampleStep * 1.7), nextSampleStep + 500);
    }

    while (my === runToken) {
      if (!running) { await sleep(120); continue; }

      const { x, y } = sampleBatch(ds.tokens, ds.filled, cfg.B, cfg.T, Math.random);
      const t0 = performance.now();
      // read the loss back every other step: the readback stalls the GPU
      // pipeline, and the free step in between keeps it full
      const readLoss = (model.step + 1) % 2 === 0;
      const loss = await model.trainStep(x, y, readLoss);
      accMs += performance.now() - t0;
      accTok += cfg.B * cfg.T;
      sessionTokens += cfg.B * cfg.T;
      if (!readLoss) continue;
      tokPerSec = tokPerSec * 0.8 + (accTok / (accMs / 1000)) * 0.2;
      accMs = 0;
      accTok = 0;
      if (my !== runToken) return;

      // exact 0.0 means the GPU silently rejected the work (a real loss can't
      // be zero); NaN means divergence. Either way this run is not salvageable.
      if (!isFinite(loss) || loss === 0) {
        console.error(`step ${model.step} loss ${loss}, restarting`);
        badRuns++;
        if (badRuns > 2) { setStatus('error', 'GPU keeps failing, giving up'); return; }
        setStatus('error', loss === 0 ? 'GPU error, restarting' : 'diverged, restarting');
        losses.length = 0;
        valPts.length = 0;
        chartReset();
        sessionTokens = 0;
        startTraining();
        return;
      }
      badRuns = 0;

      losses.push(loss);
      chartPush(loss);
      ema = losses.length === 1 ? loss : ema * 0.95 + loss * 0.05;
      $('step').textContent = model.step.toLocaleString();
      $('loss').textContent = ema.toFixed(3);
      $('lossema').textContent = ema.toFixed(3);
      $('tps').textContent = Math.round(tokPerSec).toLocaleString();
      $('mytok').textContent = sessionTokens >= 1e6
        ? `${(sessionTokens / 1e6).toFixed(1)}M`
        : sessionTokens.toLocaleString();
      if (model.step % 10 === 0) {
        $('ckptinfo').textContent = `(step ${model.step.toLocaleString()} · ${mb.toFixed(1)} MB)`;
        console.log(`step ${model.step} loss ${loss.toFixed(4)}`);
      }
      if (model.step % 6 === 0) drawChart();

      if (model.step % 500 === 0 && ds.val.length > cfg.T + 1) {
        let v = 0;
        for (let k = 0; k < 2; k++) {
          const vb = sampleBatch(ds.val, ds.val.length, cfg.B, cfg.T, Math.random);
          v += await model.valStep(vb.x, vb.y);
        }
        v /= 2;
        if (isFinite(v)) {
          valPts.push({ i: losses.length - 1, v });
          $('valloss').textContent = v.toFixed(3);
          console.log(`tabgpt: val ${v.toFixed(3)} at step ${model.step}`);
        } else {
          const tb = sampleBatch(ds.tokens, ds.filled, cfg.B, cfg.T, Math.random);
          const probe = await model.valStep(tb.x, tb.y);
          console.error(`tabgpt: non-finite val loss (${v}) at step ${model.step}; same eval on a train batch: ${probe}`);
          console.error('tabgpt debug:', await model.debugVal());
        }
      }
      // saves run after the eval so their in-flight mapping never overlaps it.
      // Bigger models save less often: each save moves the whole state blob.
      const saveEvery = model.numParams > 15_000_000 ? 1000 : 250;
      if (model.step % saveEvery === 0) saveNow?.();

      // early probe: if this machine can't push the chosen batch at a sane
      // pace, drop a tier and restart before any real progress is lost
      if (model.step === 40 && cfg.B > 4 && tokPerSec < cfg.B * cfg.T * 5) {
        batchCap = cfg.B / 2;
        console.log(`tabgpt: ${Math.round(tokPerSec)} tok/s too slow at B=${cfg.B}, retrying with B=${batchCap}`);
        startTraining();
        return;
      }

      // log-spaced samples: dense while the interesting evolution happens
      if (model.step >= nextSampleStep) {
        nextSampleStep = Math.min(Math.round(nextSampleStep * 1.7), nextSampleStep + 500);
        await doSample(model, ds);
      }
      if (pendingPrompt !== null && model.step > 100) {
        const p = pendingPrompt;
        pendingPrompt = null;
        await doPrompt(model, ds, p);
      }
    }
  } catch (err) {
    if (my !== runToken) return; // this run was torn down by a restart
    setStatus('error', 'failed to start');
    $('status').textContent = String(err);
    console.error(err);
  }
}

$('toggle').onclick = () => {
  running = !running;
  $('toggle').textContent = running ? 'pause' : 'resume';
  setStatus(running ? 'ok' : 'paused', running ? 'training' : 'paused');
};

const queuePrompt = () => {
  const input = $('prompt') as HTMLInputElement;
  pendingPrompt = input.value || input.placeholder;
  $('promptout').textContent = '…';
};
$('continue').onclick = queuePrompt;
$('prompt').addEventListener('keydown', (e) => { if ((e as KeyboardEvent).key === 'Enter') queuePrompt(); });

// best-effort save when the tab is backgrounded, reloaded or closed
document.addEventListener('visibilitychange', () => { if (document.hidden) saveNow?.(); });
window.addEventListener('pagehide', () => saveNow?.());

startTraining();
