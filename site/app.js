// Live playground: drives the real turbovec engine (compiled to WASM).
// The page generates a clustered corpus, builds an index, runs queries, and
// scores recall against brute-force ground truth computed here in JS.

import init, { WasmIndex } from './pkg/turbovec_wasm.js';

const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString('en-US');
const bytes = (b) => b >= 1e9 ? (b/1e9).toFixed(2)+' GB'
                 : b >= 1e6 ? (b/1e6).toFixed(1)+' MB'
                 : b >= 1e3 ? (b/1e3).toFixed(1)+' KB' : b.toFixed(0)+' B';

let engineReady = false;
let corpus = null;   // { data: Float32Array, clusterOf: Int32Array, n, dim }
let index = null;    // WasmIndex

// ── deterministic-ish RNG so runs are comparable within a session ──────────
let seed = 0x9e3779b9 >>> 0;
function rand() {
  seed ^= seed << 13; seed >>>= 0;
  seed ^= seed >> 17;
  seed ^= seed << 5;  seed >>>= 0;
  return (seed >>> 0) / 4294967296;
}
function gauss() { // Box–Muller
  let u = Math.max(rand(), 1e-9), v = rand();
  return Math.sqrt(-2*Math.log(u)) * Math.cos(2*Math.PI*v);
}
function normalize(vec, off, dim) {
  let s = 0; for (let i=0;i<dim;i++){ const x=vec[off+i]; s+=x*x; }
  s = Math.sqrt(s) || 1; const inv = 1/s;
  for (let i=0;i<dim;i++) vec[off+i] *= inv;
}

// ── corpus generation: a Gaussian mixture over `clusters` topic centroids ──
function generateCorpus(n, dim, clusters) {
  seed = 0x1234567 >>> 0;
  const centroids = new Float32Array(clusters*dim);
  for (let c=0;c<clusters;c++){ for (let i=0;i<dim;i++) centroids[c*dim+i]=gauss(); normalize(centroids,c*dim,dim); }
  const spread = new Float32Array(clusters);
  for (let c=0;c<clusters;c++) spread[c] = 0.05 + rand()*0.15;

  const data = new Float32Array(n*dim);
  const clusterOf = new Int32Array(n);
  for (let r=0;r<n;r++){
    const c = (rand()*clusters)|0;
    clusterOf[r]=c;
    const off=r*dim, coff=c*dim, sp=spread[c];
    for (let i=0;i<dim;i++) data[off+i] = centroids[coff+i] + gauss()*sp;
    normalize(data,off,dim);
  }
  return { data, clusterOf, centroids, spread, n, dim, clusters };
}

// ── brute-force exact top-1 (ground truth) ─────────────────────────────────
function bruteTop1(query, data, n, dim) {
  let best=-Infinity, bi=-1;
  for (let r=0;r<n;r++){
    let dot=0; const off=r*dim;
    for (let i=0;i<dim;i++) dot += query[i]*data[off+i];
    if (dot>best){best=dot;bi=r;}
  }
  return bi;
}

function log(msg){ $('log').textContent = msg; }

// ── build ──────────────────────────────────────────────────────────────────
async function buildIndex() {
  if (!engineReady) return;
  const dim = +$('dim').value;
  const n   = +$('n').value;
  const clusters = +$('cl').value;
  const bw  = +$('bw').value;
  const refine = $('refine').value;

  $('buildBtn').disabled = true; $('queryBtn').disabled = true;
  log(`Generating ${fmt(n)} vectors (dim ${dim}, ${clusters} clusters)…`);
  await new Promise(r=>setTimeout(r,10));
  corpus = generateCorpus(n, dim, clusters);

  log(`Building turbovec index (${bw}-bit, refine=${refine})…`);
  await new Promise(r=>setTimeout(r,10));
  const t0 = performance.now();
  try {
    index = new WasmIndex(dim, bw, refine === 'none' ? undefined : refine);
    // add in chunks so a huge single copy doesn't spike memory
    const CHUNK = 2000;
    for (let s=0;s<n;s+=CHUNK){
      const e=Math.min(s+CHUNK,n);
      index.add(corpus.data.subarray(s*dim, e*dim));
    }
  } catch(err){ log('Error: '+err); $('buildBtn').disabled=false; return; }
  const buildMs = performance.now()-t0;

  // memory
  const f32 = index.fp32Bytes(), idx = index.coarseBytes(), ref = index.refineBytes();
  const total = idx + ref, comp = f32/total;
  $('bars').style.display='block';
  const maxB = f32;
  $('barF32').style.width = '100%'; $('valF32').textContent = bytes(f32);
  $('barIdx').style.width = (idx/maxB*100).toFixed(1)+'%'; $('valIdx').textContent = bytes(idx);
  $('barRef').style.width = (ref/maxB*100).toFixed(1)+'%'; $('valRef').textContent = ref>0?bytes(ref):'—';
  $('refLabel').textContent = refine==='none' ? '+ refine (off)' : `+ ${refine} refine`;

  $('metrics').style.display='grid';
  $('mComp').textContent = comp.toFixed(1)+'×';
  $('mCompS').textContent = `${bytes(total)} total`;
  $('mBuild').textContent = buildMs.toFixed(0)+' ms';
  $('mBuildS').textContent = `${(n/(buildMs/1000)/1000).toFixed(0)}K vec/s`;
  $('mQuery').textContent='—'; $('mQueryS').textContent='';
  $('mRecall').textContent='—'; $('mRecallS').textContent='run queries →';

  log(`Index ready: ${fmt(n)} vectors indexed. Compression ${comp.toFixed(1)}× vs float32. Now run some queries.`);
  $('buildBtn').disabled=false; $('queryBtn').disabled=false;
}

// ── query + recall ──────────────────────────────────────────────────────────
async function runQueries() {
  if (!index || !corpus) return;
  const { data, clusterOf, centroids, spread, n, dim, clusters } = corpus;
  const k = +$('k').value;
  const rf = +$('rf').value;
  const NQ = 50;

  $('queryBtn').disabled=true;
  log(`Running ${NQ} queries (k=${k}, rerank ×${rf}) + brute-force ground truth…`);
  await new Promise(r=>setTimeout(r,10));

  let hits=0, totalMs=0, lastRows=null;
  for (let qi=0; qi<NQ; qi++){
    // query = a random cluster centroid + fresh noise (a "new" doc near a topic)
    const c = (rand()*clusters)|0;
    const q = new Float32Array(dim);
    for (let i=0;i<dim;i++) q[i] = centroids[c*dim+i] + gauss()*spread[c];
    normalize(q,0,dim);

    const truth = bruteTop1(q, data, n, dim);
    const t0=performance.now();
    const res = index.search(q, k, rf);
    totalMs += performance.now()-t0;

    const idxArr = res.indices, scArr = res.scores;
    let found=false;
    for (let j=0;j<idxArr.length;j++){ if (idxArr[j]===truth){found=true;break;} }
    if (found) hits++;

    if (qi===NQ-1){
      lastRows=[];
      for (let j=0;j<Math.min(k,idxArr.length);j++){
        const slot=idxArr[j];
        lastRows.push({rank:j+1, slot, cluster: slot>=0?clusterOf[slot]:-1,
                       score: scArr[j], truth: slot===truth});
      }
      lastRows._truth = truth; lastRows._truthCluster = clusterOf[truth];
    }
  }

  const recall = hits/NQ, avgMs = totalMs/NQ;
  $('mRecall').textContent = recall.toFixed(3);
  $('mRecallL').textContent = `Recall@${k} (true NN found)`;
  $('mRecallS').textContent = `${hits}/${NQ} queries`;
  $('mQuery').textContent = avgMs.toFixed(2)+' ms';
  $('mQueryS').textContent = `${(1000/avgMs).toFixed(0)} q/s`;

  // render last query's result table
  const body=$('resultsBody'); body.innerHTML='';
  for (const row of lastRows){
    const tr=document.createElement('tr');
    tr.innerHTML = `<td class="num">${row.rank}</td>
      <td class="num">${row.slot}</td>
      <td class="num">${row.cluster>=0?row.cluster:'—'}</td>
      <td class="num">${row.score.toFixed(4)}</td>
      <td class="${row.truth?'hit':'miss'}">${row.truth?'★ true NN':''}</td>`;
    body.appendChild(tr);
  }
  $('resultsWrap').style.display='block';
  log(`Done. Recall@${k} = ${recall.toFixed(3)} over ${NQ} queries · ${avgMs.toFixed(2)} ms/query. `+
      `Last query's true nearest neighbour was slot ${lastRows._truth} (cluster ${lastRows._truthCluster}).`);
  $('queryBtn').disabled=false;
}

// ── control labels ──────────────────────────────────────────────────────────
function wireLabels(){
  const upd=()=>{
    $('dimV').textContent=$('dim').value;
    $('nV').textContent=fmt(+$('n').value);
    $('clV').textContent=$('cl').value;
    $('bwV').textContent=$('bw').value;
    $('rfV').textContent=$('rf').value;
  };
  ['dim','n','cl','bw','rf'].forEach(id=>$(id).addEventListener('input',upd));
  upd();
}

// ── boot ────────────────────────────────────────────────────────────────────
(async function boot(){
  wireLabels();
  try {
    await init();               // loads and instantiates turbovec_wasm_bg.wasm
    engineReady=true;
    $('wasmStatus').textContent='engine ready ✓';
    $('wasmStatus').style.color='var(--ok)';
    $('buildBtn').disabled=false;
    log('WebAssembly engine loaded. Pick your parameters and press “Build index”.');
  } catch(err){
    $('wasmStatus').textContent='engine failed to load';
    $('wasmStatus').style.color='var(--bad)';
    log('Failed to load WASM module: '+err+'\nIf you opened this file directly, serve it over HTTP '+
        '(e.g. `python3 -m http.server` in the site/ directory) — ES modules need a server.');
  }
  $('buildBtn').addEventListener('click', buildIndex);
  $('queryBtn').addEventListener('click', runQueries);
})();
