"""Paints node activity (from activity.js forward passes) beside the screen.
It reads only node outputs recorded during play, never the weight files.
Usage (repo root): python3 docs/figures_src/activity_gif.py
"""
import json, os, subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'docs', 'img')
F13 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
F11 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
FAM = [
    ('Global facts', '#8b90a0', 'g0 aE F fF gt'),
    ('Move and rotate', '#f2a541', 'ovL ovR U1 U2 U3 U4 U5 U6 U7 U8 U9 U10 U11 U12 AU ovU okL okR okU mL mR mU st1 A1L A1R A1U A1S A1 P1L P1R P1S P1'),
    ('Gravity and lock', '#e0884a', 'ovD okD g mD lock st2 A2D A2S P2D P2S'),
    ('Spawn and game over', '#e5484d', 'e ASp PSp Anew Pnew ovS ge Gnew'),
    ('Stack and line clear', '#5b8def', 'LK C LKeep LSh Lnew'),
    ('Preview and random', '#7fd18b', 'h1 h2 h3 h4 h5 h6 D0 D1 D2 D3 D4 D5 D6 QK RK Qnew Rnew'),
    ('Timer', '#b48ead', 'TS TI Tnew'),
    ('Score', '#ebcb8b', 'carry SC Snew'),
    ('Fixtures', '#4c566a', 'Wc MQc MSc MTc'),
]
COL = {'#': '#3a3f4b', 'x': '#5b8def', 'a': '#f2a541', 'p': '#c47f1f', '!': '#e5484d',
       'q': '#7fd18b', 'r': '#4fa860', 'S': '#7fd18b', 'T': '#7fd18b'}
KEY = {'.': 'no key', 'l': 'LEFT', 'r': 'RIGHT', 'u': 'ROTATE', 'd': 'DOWN'}


def record(screen, keys, seed):
    out = subprocess.run(['node', 'docs/figures_src/activity.js', screen, keys, str(seed)],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def mix(c, t):
    c = c.lstrip('#'); r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    base = (29, 32, 39)
    return tuple(int(base[i] + (x - base[i]) * t) for i, x in enumerate((r, g, b)))


def frame(nodes_type, st, idx, label):
    W, H = 900, 420
    im = Image.new('RGB', (W, H), '#14161b'); d = ImageDraw.Draw(im)
    d.text((16, 10), 'Node activity: lit = output nonzero (brighter = more cells)', fill='#e7e9ee', font=F13)
    y = 40; total_nodes = 0; total_cells = 0
    for fam, col, names in FAM:
        d.text((16, y + 3), fam, fill=col, font=F11)
        x = 160
        for n in names.split():
            a = st['act'][n]; typ = nodes_type[n]
            t = 0 if a == 0 else (1.0 if typ == 'scalar' else min(1.0, 0.35 + a / 40))
            if a: total_nodes += 1; total_cells += a
            d.rectangle([x, y, x + 34, y + 18], fill=mix(col, t) if a else (29, 32, 39), outline='#2a2e38')
            d.text((x + 3, y + 3), n[:5], fill='#0e1014' if a else '#5c6170', font=F11)
            x += 37
        y += 26 + (26 if len(names.split()) > 19 else 0) * 0
        if len(names.split()) > 19:
            pass
    # screen
    rows = st['screen'].strip('\n').split('\n')
    over = rows[0] == 'GAME OVER'
    grid = rows[1:] if over else rows
    C = 13; ox, oy = 680, 40
    for yy, row in enumerate(grid):
        for xx, ch in enumerate(row):
            if ch in COL:
                d.rectangle([ox + xx * C, oy + yy * C, ox + xx * C + C - 2, oy + yy * C + C - 2], fill=COL[ch])
    d.text((16, H - 28), f'step {idx}   key: {KEY[st["key"]]}   {label}   active nodes: {total_nodes} of 87   active cells: {total_cells}',
           fill='#e7e9ee', font=F13)
    return im, total_nodes, total_cells


def main():
    # Move-and-rotate rows are long; wrap them by splitting the family.
    global FAM
    fam2 = []
    for fam, col, names in FAM:
        ns = names.split()
        if len(ns) > 13:
            for i in range(0, len(ns), 13):
                fam2.append((fam if i == 0 else '', col, ' '.join(ns[i:i + 13])))
        else:
            fam2.append((fam, col, names))
    FAM = fam2
    a = record('screen0.txt', '......', 1)
    b = record('tests/clear_T.txt', 'lr' + 'd' * 19 + '....', 1)
    c = record('tests/irot.txt', 'u..', 1)
    types = {n: t for n, t in a['nodes']}
    frames, costs = [], {}
    def label_for(st, prev_act):
        act = st['act']
        if act['ge']: return 'game over'
        if act['fF']: return 'line clear'
        if act['e'] and act['ASp']: return 'spawn'
        if act['lock']: return 'lock'
        if act['mU']: return 'rotate'
        if act['mL'] or act['mR']: return 'move'
        if act['mD'] and act['A1']: return 'fall one row'
        return 'idle tick'
    for i, st in enumerate(a['steps'] + c['steps'] + b['steps']):
        lab = label_for(st, None)
        im, n, c = frame(types, st, i + 1, lab)
        frames.append(im)
        costs.setdefault(lab, (n, c))
    frames[0].save(os.path.join(OUT, 'activity.gif'), save_all=True, append_images=frames[1:] + [frames[-1]] * 6,
                   duration=450, loop=0, optimize=True)
    print('event costs (first occurrence): label -> active nodes, active cells')
    for k, v in costs.items(): print(f'  {k}: {v[0]} nodes, {v[1]} cells')


if __name__ == '__main__':
    main()
