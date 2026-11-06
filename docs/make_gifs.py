"""Records animated GIFs of the network playing, for the README.

Each frame is a screen the network itself produced (via sim.js forward passes).
This script only paints those screens; it never reads the weight files.
Usage: python3 docs/make_gifs.py   (run from the repository root)
"""
import subprocess, os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'img')
os.makedirs(OUT, exist_ok=True)
CELL = 18
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
KEYNAME = {'.': 'no key', 'l': 'LEFT', 'r': 'RIGHT', 'u': 'ROTATE', 'd': 'DOWN'}
COL = {'#': '#3a3f4b', 'x': '#5b8def', 'a': '#f2a541', 'p': '#c47f1f', '!': '#e5484d',
       'q': '#7fd18b', 'r': '#4fa860', 'S': '#7fd18b', 'T': '#7fd18b'}


def run(screen, keys, seed=1):
    out = subprocess.run(['node', 'sim.js', screen, keys, str(seed), '1'], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    frames, cur, head = [], None, None
    for line in out.split('\n'):
        if line.startswith('step'):
            if cur: frames.append((head, cur))
            head, cur = line, []
        elif line and cur is not None:
            cur.append(line)
    if cur: frames.append((head, cur))
    return frames


def start(screen):
    rows = open(os.path.join(ROOT, screen)).read().split('\n')
    return ('step 0 key .', [r for r in rows if r])


def paint(head, rows, note=''):
    over = rows[0] == 'GAME OVER'
    grid = rows[1:] if over else rows
    w, h = 16 * CELL, 21 * CELL + 26
    im = Image.new('RGB', (w, h), '#0e1014')
    d = ImageDraw.Draw(im)
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            x0, y0 = x * CELL, y * CELL + 26
            if ch in 'smt':
                d.rectangle([x0 + 4, y0 + 4, x0 + CELL - 5, y0 + CELL - 5], outline='#2a2e38')
            if ch not in COL: continue
            inset = {'#': 0, 'S': 3, 'T': 6}.get(ch, 1)
            d.rectangle([x0 + inset, y0 + inset, x0 + CELL - 1 - inset, y0 + CELL - 1 - inset], fill=COL[ch])
            if ch == 'p':
                d.rectangle([x0 + 1, y0 + 1, x0 + CELL - 2, y0 + CELL - 2], outline='#f2a541')
    parts = head.split()
    step, key = parts[1], parts[3] if len(parts) > 3 else '.'
    label = f'step {step}   key: {KEYNAME.get(key, key)}'
    if over: label += '   GAME OVER'
    if note: label = note
    d.text((6, 6), label, fill='#e5484d' if over else '#e7e9ee', font=FONT)
    if over:
        ov = Image.new('RGB', im.size, '#e5484d')
        im = Image.blend(im, ov, 0.15)
    return im


def gif(name, frames, ms=180, hold=6):
    ims = [paint(h, r) for h, r in frames]
    ims += [ims[-1]] * hold
    ims[0].save(os.path.join(OUT, name), save_all=True, append_images=ims[1:], duration=ms, loop=0, optimize=True)
    print('wrote', name, len(ims), 'frames')


if __name__ == '__main__':
    gif('spawn.gif', [start('screen0.txt')] + run('screen0.txt', '......'), ms=600)
    gif('move.gif', run('screen0.txt', '..lllllll.rrrrrrrrrr'), ms=200)
    gif('rotate_T.gif', [start('tests/clear_T.txt')] + run('tests/clear_T.txt', 'dd.u..u..u..u..'), ms=300)
    gif('rotate_I.gif', [start('tests/irot.txt')] + run('tests/irot.txt', 'dd.u..u..u..u..rr.u..'), ms=300)
    gif('drop_lock.gif', run('screen0.txt', '..rrrr' + 'd' * 20 + '....'), ms=120)
    gif('line_clear.gif', [start('tests/clear4.txt')] + run('tests/clear4.txt', 'd' * 17 + '.' * 6), ms=220, hold=8)
    gif('timer.gif', run('screen0.txt', '.' * 18), ms=250)
    full = open(os.path.join(ROOT, 'tests', 'full_game_seed42.txt')).read().strip()
    fr = run('screen0.txt', full, 42)
    gif('full_game.gif', fr[::2] + [fr[-1]], ms=70, hold=20)
    gif('game_over.gif', run('screen0.txt', 'd' * 400, 7)[-70:], ms=110, hold=12)
