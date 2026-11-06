// Generic executor for the hand-written network. It contains no Tetris logic.
// It knows only two node types (map, scalar), one activation (clamp01),
// zero-padded "same" convolution, global sum, and scalar broadcast.
// Works in Node and in the browser.
(function (root) {
  const H = 21, W = 16;
  const CHANNELS = ['W', 'L', 'A', 'P', 'Q', 'R', 'S', 'T', 'G', 'MQ', 'MS', 'MT'];
  const SCALAR_INPUTS = ['kL', 'kR', 'kU', 'kD', 'rng'];
  // Character legend for screen files and text rendering (pure encoding).
  const LEGEND = {
    '.': [], '#': ['W'], 'x': ['L'], 'a': ['A'], 'p': ['A', 'P'],
    'q': ['Q'], 'r': ['Q', 'R'], 'm': ['MQ'], 's': ['MS'], 'S': ['MS', 'S'],
    't': ['MT'], 'T': ['MT', 'T'], '!': ['A', 'L'],
  };

  const acts = {
    clamp01: v => (v < 0 ? 0 : v > 1 ? 1 : v),
    id: v => v,
  };

  function parseGraph(text) {
    const nodes = [], outs = {};
    for (let raw of text.split('\n')) {
      const line = raw.replace(/#.*/, '').trim();
      if (!line) continue;
      const tok = line.split(/\s+/);
      if (tok[0] === 'out') { outs[tok[1]] = tok[2]; continue; }
      const [type, name, act, ...ins] = tok;
      const inputs = ins.map(s => {
        const [src, win] = s.split(':');
        if (!win) return { src, h: 0, w: 0, n: 1 };
        const [h, w] = win.split('x').map(Number);
        return { src, h, w, n: h * w };
      });
      nodes.push({ type, name, act, inputs });
    }
    return { nodes, outs };
  }

  function parseNums(text) {
    return text.trim().split(/\s+/).filter(Boolean).map(Number);
  }

  function attachWeights(graph, getFile) {
    for (const nd of graph.nodes) {
      const nums = parseNums(getFile(nd.name));
      let k = 0;
      for (const inp of nd.inputs) { inp.k = nums.slice(k, k + inp.n); k += inp.n; }
      if (nums.length !== k + 1)
        throw new Error(`weights/${nd.name}.txt: expected ${k + 1} values, got ${nums.length}`);
      nd.bias = nums[k];
    }
  }

  const zeros = () => new Float64Array(H * W);

  function conv(src, k, kh, kw, acc) {
    const ch = (kh - 1) >> 1, cw = (kw - 1) >> 1;
    for (let i = 0; i < kh; i++) for (let j = 0; j < kw; j++) {
      const wv = k[i * kw + j];
      if (wv === 0) continue;
      const dy = i - ch, dx = j - cw;
      for (let y = 0; y < H; y++) {
        const yy = y + dy; if (yy < 0 || yy >= H) continue;
        for (let x = 0; x < W; x++) {
          const xx = x + dx; if (xx < 0 || xx >= W) continue;
          acc[y * W + x] += wv * src[yy * W + xx];
        }
      }
    }
  }
  const total = m => { let s = 0; for (const v of m) s += v; return s; };

  // One forward pass: screen (channel maps) + scalar inputs -> next screen.
  function step(graph, screen, scalars) {
    const env = Object.assign({}, screen, scalars);
    for (const nd of graph.nodes) {
      const f = acts[nd.act];
      if (nd.type === 'map') {
        const acc = zeros().fill(nd.bias);
        for (const inp of nd.inputs) {
          const v = env[inp.src];
          if (v === undefined) throw new Error(`${nd.name}: unknown input ${inp.src}`);
          if (inp.h) conv(v, inp.k, inp.h, inp.w, acc);
          else for (let c = 0; c < acc.length; c++) acc[c] += inp.k[0] * v;
        }
        for (let c = 0; c < acc.length; c++) acc[c] = f(acc[c]);
        env[nd.name] = acc;
      } else {
        let s = nd.bias;
        for (const inp of nd.inputs) {
          const v = env[inp.src];
          if (v === undefined) throw new Error(`${nd.name}: unknown input ${inp.src}`);
          s += inp.k[0] * (typeof v === 'number' ? v : total(v));
        }
        env[nd.name] = f(s);
      }
    }
    const next = {};
    for (const c of CHANNELS) next[c] = env[graph.outs[c]];
    return { next, env };
  }

  function decodeScreen(text) {
    const rows = text.split('\n').filter(r => r.length);
    const scr = {}; for (const c of CHANNELS) scr[c] = zeros();
    rows.forEach((row, y) => {
      for (let x = 0; x < W; x++) {
        const ch = row[x] || '.';
        for (const c of (LEGEND[ch] || [])) scr[c][y * W + x] = 1;
      }
    });
    return scr;
  }

  function encodeScreen(scr) {
    const at = (c, i) => scr[c][i] > 0.5;
    let out = at('G', 0) ? 'GAME OVER\n' : '';
    for (let y = 0; y < H; y++) {
      let row = '';
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        let ch = '.';
        if (at('W', i)) ch = '#';
        else if (at('A', i) && at('L', i)) ch = '!';
        else if (at('P', i)) ch = 'p';
        else if (at('A', i)) ch = 'a';
        else if (at('L', i)) ch = 'x';
        else if (at('R', i)) ch = 'r';
        else if (at('Q', i)) ch = 'q';
        else if (at('S', i)) ch = 'S';
        else if (at('T', i)) ch = 'T';
        else if (at('MQ', i)) ch = 'm';
        else if (at('MS', i)) ch = 's';
        else if (at('MT', i)) ch = 't';
        row += ch;
      }
      out += row + '\n';
    }
    return out;
  }

  const api = { H, W, CHANNELS, SCALAR_INPUTS, parseGraph, attachWeights, step, decodeScreen, encodeScreen };
  if (typeof module !== 'undefined') module.exports = api; else root.WSP = api;
})(typeof window !== 'undefined' ? window : globalThis);
