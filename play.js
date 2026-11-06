// Interactive forward-pass harness: keeps the screen in state.json between calls,
// so a person (or Claude) can play one decision at a time.
// node play.js new [seed]     start a game
// node play.js <keys>         apply keys (. l r u d), print the screen after them
const fs = require('fs'), path = require('path');
const WSP = require('./engine.js');
const d = __dirname, sf = path.join(d, 'state.json');
const graph = WSP.parseGraph(fs.readFileSync(path.join(d, 'graph.txt'), 'utf8'));
WSP.attachWeights(graph, n => fs.readFileSync(path.join(d, 'weights', n + '.txt'), 'utf8'));
const arg = process.argv[2] || '';
let st;
if (arg === 'new') {
  st = { seed: Number(process.argv[3] || 1) >>> 0, steps: 0, log: '',
         scr: WSP.decodeScreen(fs.readFileSync(path.join(d, 'screen0.txt'), 'utf8')) };
} else {
  const raw = JSON.parse(fs.readFileSync(sf, 'utf8'));
  st = raw; for (const c of WSP.CHANNELS) st.scr[c] = Float64Array.from(raw.scr[c]);
  for (const k of arg) {
    st.seed = (st.seed * 1664525 + 1013904223) >>> 0;
    const sc = { kL: +(k === 'l'), kR: +(k === 'r'), kU: +(k === 'u'), kD: +(k === 'd'), rng: st.seed / 4294967296 };
    st.scr = WSP.step(graph, st.scr, sc).next; st.steps++; st.log += k;
  }
}
const out = { ...st, scr: {} };
for (const c of WSP.CHANNELS) out.scr[c] = Array.from(st.scr[c]);
fs.writeFileSync(sf, JSON.stringify(out));
console.log(`steps ${st.steps}\n` + WSP.encodeScreen(st.scr));
