"""Draws an atlas of every weight and bias in the network.

Project rule: no program may read the weight files. So this script does NOT open
weights/. Every kernel below was transcribed by hand from the values written in
those files, and the script only draws what is typed here.
Usage (repo root): python3 docs/figures_src/weight_atlas.py
"""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F12 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 12)
F11 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
F10 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 10)
F16 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)


def row(*v): return [list(v)]
def col(*v): return [[x] for x in v]
S = lambda v: [[v]]          # a scalar input (one weight)
E9 = 1e9

# 3x3 and 5x5 one-hot kernels, typed as (row, col) of the single 1
def hot(n, r, c):
    k = [[0] * n for _ in range(n)]; k[r][c] = 1; return k

I_ = [[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,0],[0,0,0,0,0],[0,0,0,0,0]]
O_ = [[0,0,0,0,0],[0,0,0,0,0],[0,1,1,0,0],[0,1,1,0,0],[0,0,0,0,0]]
T_ = [[0,0,0,0,0],[0,0,0,0,0],[0,1,1,1,0],[0,0,1,0,0],[0,0,0,0,0]]
S_ = [[0,0,0,0,0],[0,0,0,0,0],[0,0,1,1,0],[0,1,1,0,0],[0,0,0,0,0]]
Z_ = [[0,0,0,0,0],[0,0,0,0,0],[0,1,1,0,0],[0,0,1,1,0],[0,0,0,0,0]]
J_ = [[0,0,0,0,0],[0,0,0,0,0],[0,1,1,1,0],[0,0,0,1,0],[0,0,0,0,0]]
L_ = [[0,0,0,0,0],[0,0,0,0,0],[0,1,1,1,0],[0,1,0,0,0],[0,0,0,0,0]]
SHIFT8 = row(*([0] * 16 + [1]))

# node: (family, [(input label, kernel)], bias)
NODES = [
 ('Global facts', 'g0', [('G', S(1))], 0),
 ('Global facts', 'aE', [('A', S(-1))], 1),
 ('Global facts', 'F', [('L', row(*[1] * 19))], -9),
 ('Global facts', 'fF', [('F', S(1))], 0),
 ('Global facts', 'gt', [('T', S(-1))], 1),

 ('Move and rotate', 'ovL', [('A', row(0,0,1)), ('W', S(1)), ('L', S(1))], -1),
 ('Move and rotate', 'ovR', [('A', row(1,0,0)), ('W', S(1)), ('L', S(1))], -1),
 ('Move and rotate', 'U1', [('P', hot(3,0,1)), ('A', hot(3,0,2))], -1),
 ('Move and rotate', 'U2', [('P', hot(3,1,2)), ('A', hot(3,2,2))], -1),
 ('Move and rotate', 'U3', [('P', hot(3,2,1)), ('A', hot(3,2,0))], -1),
 ('Move and rotate', 'U4', [('P', hot(3,1,0)), ('A', hot(3,0,0))], -1),
 ('Move and rotate', 'U5', [('P', hot(3,0,2)), ('A', hot(5,2,4))], -1),
 ('Move and rotate', 'U6', [('P', hot(3,2,2)), ('A', hot(5,4,2))], -1),
 ('Move and rotate', 'U7', [('P', hot(3,2,0)), ('A', hot(5,2,0))], -1),
 ('Move and rotate', 'U8', [('P', hot(3,0,0)), ('A', hot(5,0,2))], -1),
 ('Move and rotate', 'U9', [('P', hot(5,0,2)), ('A', hot(5,0,4))], -1),
 ('Move and rotate', 'U10', [('P', hot(5,2,4)), ('A', hot(5,4,4))], -1),
 ('Move and rotate', 'U11', [('P', hot(5,4,2)), ('A', hot(5,4,0))], -1),
 ('Move and rotate', 'U12', [('P', hot(5,2,0)), ('A', hot(5,0,0))], -1),
 ('Move and rotate', 'AU', [('P', S(1))] + [('U%d' % i, S(1)) for i in range(1, 13)], 0),
 ('Move and rotate', 'ovU', [('AU', S(1)), ('W', S(1)), ('L', S(1))], -1),
 ('Move and rotate', 'okL', [('ovL', S(-1))], 1),
 ('Move and rotate', 'okR', [('ovR', S(-1))], 1),
 ('Move and rotate', 'okU', [('AU', S(1)), ('ovU', S(-10))], -3),
 ('Move and rotate', 'mL', [('kL', S(1)), ('okL', S(1)), ('g0', S(-1))], -1),
 ('Move and rotate', 'mR', [('kR', S(1)), ('okR', S(1)), ('g0', S(-1))], -1),
 ('Move and rotate', 'mU', [('kU', S(1)), ('okU', S(1)), ('g0', S(-1))], -1),
 ('Move and rotate', 'st1', [('mL', S(-1)), ('mR', S(-1)), ('mU', S(-1))], 1),
 ('Move and rotate', 'A1L', [('A', row(0,0,1)), ('mL', S(1))], -1),
 ('Move and rotate', 'A1R', [('A', row(1,0,0)), ('mR', S(1))], -1),
 ('Move and rotate', 'A1U', [('AU', S(1)), ('mU', S(1))], -1),
 ('Move and rotate', 'A1S', [('A', S(1)), ('st1', S(1))], -1),
 ('Move and rotate', 'A1', [('A1L', S(1)), ('A1R', S(1)), ('A1U', S(1)), ('A1S', S(1))], 0),
 ('Move and rotate', 'P1L', [('P', row(0,0,1)), ('mL', S(1))], -1),
 ('Move and rotate', 'P1R', [('P', row(1,0,0)), ('mR', S(1))], -1),
 ('Move and rotate', 'P1S', [('P', S(1)), ('mU', S(1)), ('st1', S(1))], -1),
 ('Move and rotate', 'P1', [('P1L', S(1)), ('P1R', S(1)), ('P1S', S(1))], 0),

 ('Gravity and lock', 'ovD', [('A1', col(1,0,0)), ('W', S(1)), ('L', S(1))], -1),
 ('Gravity and lock', 'okD', [('ovD', S(-1))], 1),
 ('Gravity and lock', 'g', [('gt', S(1)), ('kD', S(1)), ('g0', S(-2))], 0),
 ('Gravity and lock', 'mD', [('g', S(1)), ('okD', S(1))], -1),
 ('Gravity and lock', 'lock', [('g', S(1)), ('okD', S(-1))], 0),
 ('Gravity and lock', 'st2', [('mD', S(-1)), ('lock', S(-1))], 1),
 ('Gravity and lock', 'A2D', [('A1', col(1,0,0)), ('mD', S(1))], -1),
 ('Gravity and lock', 'A2S', [('A1', S(1)), ('st2', S(1))], -1),
 ('Gravity and lock', 'P2D', [('P1', col(1,0,0)), ('mD', S(1))], -1),
 ('Gravity and lock', 'P2S', [('P1', S(1)), ('st2', S(1))], -1),

 ('Spawn and game over', 'e', [('aE', S(1)), ('fF', S(-1)), ('g0', S(-1))], 0),
 ('Spawn and game over', 'ASp', [('Q', SHIFT8), ('e', S(1))], -1),
 ('Spawn and game over', 'PSp', [('R', SHIFT8), ('e', S(1))], -1),
 ('Spawn and game over', 'Anew', [('A2D', S(1)), ('A2S', S(1)), ('ASp', S(1))], 0),
 ('Spawn and game over', 'Pnew', [('P2D', S(1)), ('P2S', S(1)), ('PSp', S(1))], 0),
 ('Spawn and game over', 'ovS', [('Q', SHIFT8), ('L', S(1)), ('e', S(1))], -2),
 ('Spawn and game over', 'ge', [('ovS', S(1))], 0),
 ('Spawn and game over', 'Gnew', [('g0', S(1)), ('ge', S(1))], 0),

 ('Stack and line clear', 'LK', [('A1', S(1)), ('lock', S(1))], -1),
 ('Stack and line clear', 'C', [('F', col(*([0] * 19 + [1] * 20)))], 0),
 ('Stack and line clear', 'LKeep', [('L', S(1)), ('C', S(-1))], 0),
 ('Stack and line clear', 'LSh', [('L', col(1,0,0)), ('C', S(1))], -1),
 ('Stack and line clear', 'Lnew', [('LKeep', S(1)), ('LSh', S(1)), ('LK', S(1))], 0),

 ('Preview and random', 'h1', [('rng', S(7 * E9))], -1 * E9),
 ('Preview and random', 'h2', [('rng', S(7 * E9))], -2 * E9),
 ('Preview and random', 'h3', [('rng', S(7 * E9))], -3 * E9),
 ('Preview and random', 'h4', [('rng', S(7 * E9))], -4 * E9),
 ('Preview and random', 'h5', [('rng', S(7 * E9))], -5 * E9),
 ('Preview and random', 'h6', [('rng', S(7 * E9))], -6 * E9),
 ('Preview and random', 'D0', [('MQ', S(1)), ('h1', S(-1)), ('e', S(1))], -1),
 ('Preview and random', 'D1', [('MQ', S(1)), ('h1', S(1)), ('h2', S(-1)), ('e', S(1))], -2),
 ('Preview and random', 'D2', [('MQ', S(1)), ('h2', S(1)), ('h3', S(-1)), ('e', S(1))], -2),
 ('Preview and random', 'D3', [('MQ', S(1)), ('h3', S(1)), ('h4', S(-1)), ('e', S(1))], -2),
 ('Preview and random', 'D4', [('MQ', S(1)), ('h4', S(1)), ('h5', S(-1)), ('e', S(1))], -2),
 ('Preview and random', 'D5', [('MQ', S(1)), ('h5', S(1)), ('h6', S(-1)), ('e', S(1))], -2),
 ('Preview and random', 'D6', [('MQ', S(1)), ('h6', S(1)), ('e', S(1))], -2),
 ('Preview and random', 'QK', [('Q', S(1)), ('e', S(-1))], 0),
 ('Preview and random', 'RK', [('R', S(1)), ('e', S(-1))], 0),
 ('Preview and random', 'Qnew', [('QK', S(1)), ('D0 I', I_), ('D1 O', O_), ('D2 T', T_), ('D3 S', S_),
                                 ('D4 Z', Z_), ('D5 J', J_), ('D6 L', L_)], 0),
 ('Preview and random', 'Rnew', [('RK', S(1))] + [('D%d' % i, S(1)) for i in (0, 2, 3, 4, 5, 6)], 0),

 ('Timer', 'TS', [('T', col(1,0,0)), ('MT', S(1))], -1),
 ('Timer', 'TI', [('MT', col(-1,1,0)), ('gt', S(1))], -1),
 ('Timer', 'Tnew', [('TS', S(1)), ('TI', S(1))], 0),

 ('Score', 'carry', [('S', col(*([0] * 16 + [1] * 15))), ('MS', col(*([0] * 15 + [1] + [-1] * 15))), ('fF', S(1))], -1),
 ('Score', 'SC', [('S', S(1)), ('carry', S(1))], -1),
 ('Score', 'Snew', [('S', S(1)), ('carry', S(1)), ('SC', S(-2))], 0),

 ('Fixtures', 'Wc', [('W', S(1))], 0),
 ('Fixtures', 'MQc', [('MQ', S(1))], 0),
 ('Fixtures', 'MSc', [('MS', S(1))], 0),
 ('Fixtures', 'MTc', [('MT', S(1))], 0),
]

FAMCOL = {'Global facts': '#8b90a0', 'Move and rotate': '#f2a541', 'Gravity and lock': '#e0884a',
          'Spawn and game over': '#e5484d', 'Stack and line clear': '#5b8def', 'Preview and random': '#7fd18b',
          'Timer': '#b48ead', 'Score': '#ebcb8b', 'Fixtures': '#4c566a'}


def color(v):
    if v == 0: return (40, 44, 54)
    if v >= 1e8: return (127, 209, 139)
    if v <= -1e8: return (60, 140, 80)
    if v > 0: return (242, 165, 65) if v == 1 else (255, 214, 120)
    return (91, 141, 239) if v == -1 else ((150, 110, 230) if v == -2 else (229, 72, 77))


def count():
    w = b = 0
    for _, _, ins, _ in NODES:
        for _, k in ins: w += sum(len(r) for r in k)
        b += 1
    return w, b


def draw():
    C = 8  # cell size
    W = 1500
    # layout pass: each node is a block placed in a flowing grid per family
    blocks = []
    for fam, name, ins, bias in NODES:
        parts, wsum, hmax = [], 0, 0
        for lab, k in ins:
            kw, kh = len(k[0]) * C, len(k) * C
            kw = max(kw, 7 * len(lab) + 2)
            parts.append((lab, k, kw, kh)); wsum += kw + 8; hmax = max(hmax, kh)
        wsum += 46  # bias
        blocks.append((fam, name, parts, bias, max(wsum + 10, 70), hmax + 44))
    y, x, rowh, cur = 70, 20, 0, None
    pos = []
    for blk in blocks:
        fam, name, parts, bias, bw, bh = blk
        if fam != cur:
            if cur is not None: y += rowh + 34
            x, rowh, cur = 20, 0, fam
            pos.append(('fam', fam, 20, y)); y += 22
        if x + bw > W - 20: x = 20; y += rowh + 10; rowh = 0
        pos.append(('node', blk, x, y)); x += bw + 10; rowh = max(rowh, bh)
    H = y + rowh + 120
    im = Image.new('RGB', (W, H), '#14161b'); d = ImageDraw.Draw(im)
    nw, nb = count()
    d.text((20, 18), f'Every weight and bias in the network ({nw} weights + {nb} biases = {nw + nb} values), transcribed by hand from weights/',
           fill='#e7e9ee', font=F16)
    for item in pos:
        if item[0] == 'fam':
            _, fam, fx, fy = item
            d.text((fx, fy), fam, fill=FAMCOL[fam], font=F12)
            continue
        _, (fam, name, parts, bias, bw, bh), bx, by = item
        d.rounded_rectangle([bx, by, bx + bw, by + bh], 6, fill='#1d2027', outline=FAMCOL[fam])
        d.text((bx + 6, by + 4), name, fill=FAMCOL[fam], font=F12)
        px = bx + 6
        for lab, k, kw, kh in parts:
            d.text((px, by + 20), lab, fill='#8b90a0', font=F10)
            for r, rr in enumerate(k):
                for c, v in enumerate(rr):
                    x0, y0 = px + c * C, by + 34 + r * C
                    d.rectangle([x0, y0, x0 + C - 2, y0 + C - 2], fill=color(v))
            px += kw + 8
        d.text((px, by + 20), 'bias', fill='#8b90a0', font=F10)
        bl = ('%g' % bias) if abs(bias) < 1e8 else '%de9' % round(bias / 1e9)
        d.rectangle([px, by + 34, px + C - 2, by + 34 + C - 2], fill=color(bias))
        d.text((px + C + 2, by + 31), bl, fill='#e7e9ee', font=F10)
    # legend
    ly = H - 60
    items = [(1, '+1'), (-1, '-1'), (-2, '-2'), (-10, '-3, -9, -10'), (2, 'other positive'), (7e9, '7e9 slope'),
             (-3e9, '-k e9 threshold'), (0, '0')]
    lx = 20
    for v, t in items:
        d.rectangle([lx, ly, lx + 14, ly + 14], fill=color(v)); d.text((lx + 20, ly), t, fill='#e7e9ee', font=F11)
        lx += 40 + 7 * len(t) + 20
    d.text((20, ly + 26), 'Each block is one node. Small grids are convolution kernels (one cell per weight, laid out as in the file); single cells are scalar weights; the last cell is the bias.',
           fill='#8b90a0', font=F11)
    im.save(os.path.join(ROOT, 'docs', 'img', 'weight_atlas.png'), optimize=True)
    print('wrote weight_atlas.png', im.size, 'weights', nw, 'biases', nb)


if __name__ == '__main__':
    draw()
