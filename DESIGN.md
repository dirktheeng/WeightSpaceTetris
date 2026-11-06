# Tetris in Weight Space: Design Note

This is a complete game of Tetris on a standard 10×20 board. All of its rules live in 87 hand-written weight files, 1,035 numbers in total. Nothing was trained or generated. Each value was decided by reasoning and typed into a text file. The network was checked by stepping it (forward passes) and reading the output screens. No weight ever needed changing; the only defects found were in the host code and in one test file.

## Files

| File | Role |
|---|---|
| `graph.txt` | Architecture: node names, input wiring and kernel window sizes. No values. |
| `weights/<node>.txt` | The values for one node: kernels in input order, then the bias. Only numbers. |
| `screen0.txt` | The starting screen, drawn as characters (walls and fixtures). |
| `engine.js` | Generic executor: zero-padded convolution, global sum, broadcast, clamp. No game logic. |
| `play.template.html`, `build.js` | Browser front end; `build.js` copies the files into `play.html` verbatim. |
| `sim.js` | Debug harness: runs forward passes on a key script and prints the screens. |
| `tests/*.txt` | Hand-drawn start screens used to exercise line clears and I rotation. |

## The framework (what it does and does not do)

Every 100 ms the page delivers at most one queued key as four binary scalars (`kL`, `kR`, `kU`, `kD`) plus one uniform random number `rng`. It runs one forward pass and draws the returned screen, which is then fed back in unchanged. It never checks collisions, clears lines, picks pieces, counts score or edits the output. It only paints the planes it receives, so the score shows as the network's own column of binary cells. `AUDIT.md` reviews all of the framework code.

The engine knows two node types:

* **map node**: output is a 21×16 map. Each map input is convolved with an h×w kernel (centered, zero padded). Each scalar input is added to every cell with one weight.
* **scalar node**: output is one number. Each map input contributes its global sum times one weight. Each scalar input contributes its value times one weight.

Every node adds a bias and applies the same activation.

## Why one activation: clamp01

`clamp01(v) = min(1, max(0, v))`. Every value on the screen is 0 or 1, and every weight is a small integer, so every pre-activation is an integer. On integers, clamp01 is an exact logic gate:

* AND of n inputs: weights 1, bias `-(n-1)`.
* OR: weights 1, bias 0 (the clamp caps the sum at 1).
* NOT x: weight -1, bias 1.
* "at least k of these": weights 1, bias `-(k-1)`.

Plain ReLU would let ORs grow past 1, which would need extra layers to renormalize. A sigmoid would never give exact 0 and 1, so errors would build up over thousands of steps. clamp01 keeps the state binary forever. The one non-binary input, `rng`, is turned into a sharp step with a slope of 7×10⁹, so it becomes binary too.

## Why convolution (weight sharing)

Tetris rules are the same at every board position. A shifted copy of the piece, an overlap test or a row test is the same small stencil everywhere, so it is written once as a kernel. A dense network would need the same rule repeated for all 336 cells. That would be roughly 100 times more values, and too many to write or read by hand.

## The screen (the only state)

The screen is 21 rows by 16 columns, with 12 binary channels, like the color planes of an image.

```
cols 0..11: board          cols 12..15: status
#..........#.qq.   row 0    preview box (rows 0-1)
#..........#qrq.   row 1    'm' anchor at (13,1)
...                         score bits: column 13, rows 4-19 (high bit on top)
#..........#.s.t            timer cells: column 15, rows 4-10
############....   row 20   floor
```

| Channel | Meaning | Why it is needed |
|---|---|---|
| `W` | walls and floor (fixture) | Collisions treat walls like locked cells, so no special edge cases. It is kept apart from `L` so walls never count toward a full row. |
| `L` | locked cells | the stack |
| `A` | falling piece cells | moved, rotated and tested separately from the stack |
| `P` | pivot of the falling piece | Rotation needs a center. The O piece has no pivot, so it can't rotate. |
| `Q`, `R` | preview piece cells and its pivot | Spawning is a plain copy of the preview into the board. |
| `S` | score bits (binary count of cleared lines) | a 16-bit counter |
| `T` | timer token | gravity clock |
| `G` | game over (the whole map is lit) | freezes play |
| `MQ`, `MS`, `MT` | fixtures: preview anchor, score field, timer field | tell position-independent kernels where those fields are |

Fixtures are copied through by identity nodes (`Wc`, `MQc`, `MSc`, `MTc`: weight 1, bias 0). That way the network, not the framework, produces the whole next screen.

## The step, stage by stage

### Global facts (`g0`, `aE`, `F`, `fF`, `gt`)

* `g0` = any `G` lit: game over. Every action gate later subtracts it.
* `aE` = NOT any `A`: there is no falling piece.
* `F` (map, kernel 1×19 of ones, bias -9): a cell is 1 when 10 locked cells fall within ±9 columns of it. Only the ten interior cells of a full row reach 10. Wall columns and status columns can see at most 9 interior cells, so they can never fire.
* `fF` = any `F`: some row is full.
* `gt` = NOT any `T`: the timer token has left its field, so this step is a gravity tick.

### Stage 1: player move

* `ovL`, `ovR`: kernel `0 0 1` reads the cell to the right, so it gives the piece shifted left (`1 0 0` gives the piece shifted right). Each is ANDed (bias -1) with walls plus locked cells. A 1 anywhere means that move collides.
* **Rotation, `U1`..`U12`.** Clockwise rotation in screen coordinates (y down) maps an offset d = (dx,dy) from the pivot to Rd = (-dy, dx). Each node handles one offset d. It fires at cell x when the pivot sits at x - Rd AND the piece has a cell at x - Rd + d. It has two one-hot kernels (pivot at -Rd, piece at d - Rd) and bias -1, so the rotated cell lands in place directly. Twelve offsets cover every shape: the 8 neighbors (`U1`..`U4` sides, `U5`..`U8` diagonals) and the 4 cells at distance 2 that only the I piece uses (`U9`..`U12`). Side offsets fit in 3×3 windows; diagonal and distance-2 offsets need 5×5.
  | node | d | pivot kernel at | piece kernel at |
  |---|---|---|---|
  | U1 | (1,0) | (0,-1) | (1,-1) |
  | U2 | (0,1) | (1,0) | (1,1) |
  | U3 | (-1,0) | (0,1) | (-1,1) |
  | U4 | (0,-1) | (-1,0) | (-1,-1) |
  | U5 | (1,1) | (1,-1) | (2,0) |
  | U6 | (-1,1) | (1,1) | (0,2) |
  | U7 | (-1,-1) | (-1,1) | (-2,0) |
  | U8 | (1,-1) | (-1,-1) | (0,-2) |
  | U9 | (2,0) | (0,-2) | (2,-2) |
  | U10 | (0,2) | (2,0) | (2,2) |
  | U11 | (-2,0) | (0,2) | (-2,2) |
  | U12 | (0,-2) | (-2,0) | (-2,-2) |

  (Kernel positions are (dx,dy); grid row = dy + half height, column = dx + half width.)
* `AU` = OR of the pivot (which stays put) and the twelve `U` maps: the rotated piece.
* `ovU`: rotated piece AND (wall or locked).
* `okL`, `okR` = NOT any overlap. `okU` = clamp(cells in `AU` - 3 - 10 × overlaps). This is 1 only when the rotated piece still has all 4 cells and touches nothing. That one test also rejects O (no pivot, so `AU` is empty) and any rotation pushed off the top of the screen.
* `mL`, `mR`, `mU` = key AND ok AND NOT game over (weights 1, 1, -1, bias -1). `st1` = none of them.
* `A1L`, `A1R`, `A1U`, `A1S` gate each candidate piece by its scalar (bias -1). `A1` ORs them. The pivot follows the same path (`P1L`, `P1R`, `P1S`, `P1`); a rotation keeps the pivot, so `P1S` passes on either rotate or stay.

### Stage 2: gravity and locking

* `ovD`: `A1` shifted down one row (vertical kernel `1 0 0`, top entry) AND solid. `okD` = no overlap.
* `g` = (timer tick OR down key) AND NOT game over. Weight -2 on `g0` cancels even when both inputs are 1.
* `mD` = g AND okD: fall one row. `lock` = g AND NOT okD: the piece can't fall, so it locks. `st2` = neither.
* `A2D`, `A2S`, `P2D`, `P2S` gate the fallen or kept piece; a locking piece goes into neither.

### Stage 3: spawn and game over

* `e` = no falling piece AND no full row AND NOT game over. Spawning waits until every full row has been cleared.
* `ASp`, `PSp`: kernel 1×17 with its single 1 at the far right reads 8 columns to the right. That copies the preview box (columns 12..15) onto the spawn spot (columns 4..7, pivot at column 5), gated by `e`.
* `Anew`, `Pnew` OR the fallen, kept and spawned pieces into the next `A` and `P`.
* `ovS`: copied preview AND locked AND e (bias -2). `ge` = any of it. `Gnew` = old game over OR new one, broadcast to the whole map. The overlapping piece is still drawn, so the player sees what topped out. Once `G` is set, every action gate is 0 and play freezes.

### Locked cells and line clear (`LK`, `C`, `LKeep`, `LSh`, `Lnew`)

* `LK` = `A1` AND lock: the piece's cells join the stack.
* `C` (kernel 39×1, ones at offsets 0..+19): a cell is 1 when some full row is at its row or below. Those are the rows that must move down.
* `LKeep` = L AND NOT C. `LSh` = cell above AND C. `Lnew` = OR of the three.

One row is removed per step. With several full rows, everything above the lowest one drops by one, and the next step removes the next one. This is why spawning waits for `fF` = 0. It also makes the score exact, since the network adds 1 per step while a full row exists.

### New preview (`h1`..`h6`, `D0`..`D6`, `QK`, `RK`, `Qnew`, `Rnew`)

* `h_k` = step(rng ≥ k/7): weight 7×10⁹, bias -k×10⁹. The values were chosen to stay round.
* `D_t` = anchor AND (h_t AND NOT h_{t+1}) AND e. That marks the anchor cell for exactly one piece t, only on a spawn step. The first and last pieces drop the missing h term, and their bias is adjusted to match.
* `Qnew` = kept preview (`QK` = Q AND NOT e) OR each `D_t` convolved with its 5×5 shape kernel. A kernel is the shape mirrored through its center, because convolution reads `in(x + k)`. To stamp a cell at offset c, the 1 goes at -c.
* `Rnew` = kept pivot OR every `D_t` except `D1` (O has no pivot).

Piece order and shapes (offsets from the pivot, y up is negative). Every piece spawns flat, in two rows:

| t | piece | cells | pivot choice |
|---|---|---|---|
| 0 | I | (-1,0) (0,0) (1,0) (2,0) | second cell; rotates about a cell like classic NES Tetris |
| 1 | O | (0,0) (1,0) (0,-1) (1,-1) | none; O never rotates |
| 2 | T | (-1,0) (0,0) (1,0) (0,-1) | center |
| 3 | S | (-1,0) (0,0) (0,-1) (1,-1) | bottom-center |
| 4 | Z | (-1,-1) (0,-1) (0,0) (1,0) | bottom-center |
| 5 | J | (-1,-1) (-1,0) (0,0) (1,0) | center of the bar |
| 6 | L | (1,-1) (-1,0) (0,0) (1,0) | center of the bar |

These pivots keep every cell of every rotation within distance 2 of the pivot, which needs exactly 12 rotation detectors. The I piece is the only one that uses distance 2.

### Gravity timer (`TS`, `TI`, `Tnew`)

The timer field is 7 cells in column 15. `TS` moves the token down one cell (kernel `1 0 0`) and masks it to the field, so it drops out after the last cell. `TI` places a new token at the field's first cell (field here AND NOT field above, kernel `-1 1 0`) on the step when the field is empty (`gt`). One cycle is 8 steps, or 0.8 s at 10 steps per second, close to classic level 0.

### Score (`carry`, `SC`, `Snew`)

The score is a binary counter of cleared lines, low bit at the bottom (row 19). It increments when `fF` is 1.

* `carry` (31×1 kernels on `S` and `MS`): fires at a score cell when increment AND every lower bit is 1. The sum of (bit - field) over the 15 cells below is 0 only when all of them are 1, and negative otherwise.
* `SC` = S AND carry. `Snew` = S + carry - 2×SC = S XOR carry.

## Compactness

87 nodes and 1,035 numbers. Only 14 distinct values appear: 0, 1, -1, -2, -3, -9, -10, 7×10⁹ and -1×10⁹..-6×10⁹. Most files hold 2 to 6 numbers. The large files are mostly zeros in long kernels: `F` (19), `C` (39), `carry` (62), the three spawn copies (17 each), and `Qnew` (seven 5×5 shapes).

## Index of every weight file

| File | Purpose |
|---|---|
| `g0` | game over is set (any G cell) |
| `aE` | no falling piece |
| `F` | cell lies in a full row |
| `fF` | some row is full |
| `gt` | gravity tick (timer field empty) |
| `ovL`, `ovR` | overlap if the piece moved left / right |
| `U1`..`U12` | rotation: one detector per offset from the pivot |
| `AU` | rotated piece |
| `ovU` | overlap of the rotated piece |
| `okL`, `okR`, `okU` | move left / right / rotate is legal |
| `mL`, `mR`, `mU` | do that move: key AND legal AND not game over |
| `st1` | no move this step |
| `A1L`, `A1R`, `A1U`, `A1S` | gated candidate pieces |
| `A1` | piece after the player's move |
| `P1L`, `P1R`, `P1S`, `P1` | pivot after the player's move |
| `ovD` | overlap if the piece falls |
| `okD` | falling is legal |
| `g` | gravity applies this step (tick or down key, not game over) |
| `mD` | fall one row |
| `lock` | lock the piece |
| `st2` | neither fall nor lock |
| `A2D`, `A2S`, `P2D`, `P2S` | piece and pivot after gravity |
| `e` | spawn this step |
| `ASp`, `PSp` | preview copied into the spawn spot |
| `Anew`, `Pnew` | next falling piece and pivot (output) |
| `ovS` | spawned piece overlaps the stack |
| `ge` | game over happens now |
| `Gnew` | next game-over plane (output) |
| `LK` | cells added by locking |
| `C` | cell is at or above a full row |
| `LKeep` | locked cells that stay |
| `LSh` | locked cells that move down one row |
| `Lnew` | next locked plane (output) |
| `h1`..`h6` | rng ≥ k/7 steps |
| `D0`..`D6` | anchor marked for piece I, O, T, S, Z, J, L on a spawn step |
| `QK`, `RK` | preview kept when no spawn happens |
| `Qnew`, `Rnew` | next preview cells and pivot (output) |
| `TS` | timer token moves down |
| `TI` | timer token re-inserted at the start |
| `Tnew` | next timer plane (output) |
| `carry` | binary carry at each score cell |
| `SC` | score bit AND carry |
| `Snew` | next score bits (output) |
| `Wc`, `MQc`, `MSc`, `MTc` | fixtures copied through unchanged |

## How it was checked

All checks used forward passes through `sim.js` with key scripts, with the printed screens read by eye:
* moving into both walls stops the piece; soft drop, locking and respawn work;
* T and I rotate through all four states, and an I rotation into the right wall is refused;
* for each of the seven pieces, a test screen `tests/clear_<piece>.txt` has a bottom row full except exactly that piece's footprint. Each piece was moved left and right, then soft-dropped. In every case the row cleared, the piece's remaining cells dropped one row, and the score went from 0 to 1. Run them with `node sim.js tests/clear_T.txt "lr$(printf 'd%.0s' {1..19})...." 1 1`;
* four lines clear one per step with the stack shifting down, and the score bits count up in binary;
* holding drop with no moves tops out the board, shows the overlap, and freezes all input;
* sweeping `rng` across its seven bins produces I, O, T, S, Z, J, L in the preview;
* a complete game was played one decision at a time with `play.js` (seed 42), from the empty start screen to game over in 512 steps. Each move was chosen from the printed screen. The game placed all seven pieces from spawn to lock, including rotated J, I, S, L and T placements. It cleared a line, incremented the score bit and ended in a top-out. The full key record is `tests/full_game_seed42.txt`; replay it with `node sim.js screen0.txt "$(cat tests/full_game_seed42.txt)" 42 1`.
