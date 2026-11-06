// Forward-pass harness: runs the network on a key script and prints output screens.
// Usage: node sim.js <screen file> <keys> [seed] [print-every]
// keys: one char per step: . none, l left, r right, u rotate, d down
const fs = require('fs'), path = require('path');
const WSP = require('./engine.js');
const dir = __dirname;
const graph = WSP.parseGraph(fs.readFileSync(path.join(dir, 'graph.txt'), 'utf8'));
WSP.attachWeights(graph, n => fs.readFileSync(path.join(dir, 'weights', n + '.txt'), 'utf8'));

const [, , scrFile = 'screen0.txt', keys = '', seedArg = '1', every = '1'] = process.argv;
let seed = Number(seedArg) >>> 0;
const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };

let scr = WSP.decodeScreen(fs.readFileSync(path.join(dir, scrFile), 'utf8'));
for (let t = 0; t < keys.length; t++) {
  const k = keys[t];
  const sc = { kL: +(k === 'l'), kR: +(k === 'r'), kU: +(k === 'u'), kD: +(k === 'd'), rng: rand() };
  scr = WSP.step(graph, scr, sc).next;
  if ((t + 1) % Number(every) === 0 || t === keys.length - 1)
    console.log(`step ${t + 1} key ${k}\n` + WSP.encodeScreen(scr));
}
