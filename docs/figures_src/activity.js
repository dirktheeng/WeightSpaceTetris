// Records node ACTIVITY (outputs during forward passes) for the README figures.
// It never reads or interprets weight values: it only counts how many cells of
// each node's output are nonzero at each step.
// Usage: node docs/figures_src/activity.js <screen> <keys> <seed>  > out.json
const fs = require('fs'), path = require('path');
const root = path.join(__dirname, '..', '..');
const WSP = require(path.join(root, 'engine.js'));
const graph = WSP.parseGraph(fs.readFileSync(path.join(root, 'graph.txt'), 'utf8'));
WSP.attachWeights(graph, n => fs.readFileSync(path.join(root, 'weights', n + '.txt'), 'utf8'));
const [, , scrFile, keys, seedArg = '1'] = process.argv;
let seed = Number(seedArg) >>> 0;
const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
let scr = WSP.decodeScreen(fs.readFileSync(path.join(root, scrFile), 'utf8'));
const steps = [];
for (const k of keys) {
  const sc = { kL: +(k === 'l'), kR: +(k === 'r'), kU: +(k === 'u'), kD: +(k === 'd'), rng: rand() };
  const { next, env } = WSP.step(graph, scr, sc);
  const act = {};
  for (const nd of graph.nodes) {
    const v = env[nd.name];
    act[nd.name] = typeof v === 'number' ? (v > 0 ? 1 : 0) : v.reduce((s, x) => s + (x > 0 ? 1 : 0), 0);
  }
  scr = next;
  steps.push({ key: k, act, screen: WSP.encodeScreen(scr) });
}
console.log(JSON.stringify({ nodes: graph.nodes.map(n => [n.name, n.type]), steps }));
