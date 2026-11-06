# WeightSpaceTetris

**Tetris implemented as a hand-programmed convolutional threshold network. No training, no weight generators: 87 nodes, 1,035 hand-written parameters.**

<p align="center"><img src="docs/img/full_game.gif" alt="A full game played on the network, from empty board to game over" width="300"></p>

Every frame above is a forward pass of the network. The game state is a 12-channel binary image. The network maps (state, key, random number) to the next state, and the host loop feeds each output back in as the next input. Every weight was derived by reasoning and written into a plain-text file; there is no optimizer, compiler or code generator anywhere in the pipeline.

**Play:** clone the repository and open `play.html`. Arrow keys move, rotate (up) and soft drop (down); P pauses and N starts a new game.

---

## Contents

1. [Problem statement and constraints](#1-problem-statement-and-constraints)
2. [Related work](#2-related-work)
3. [System overview](#3-system-overview)
4. [Computational model](#4-computational-model)
5. [State representation](#5-state-representation)
6. [Network walkthrough](#6-network-walkthrough)
7. [Host framework](#7-host-framework)
8. [Design decisions](#8-design-decisions)
9. [Verification](#9-verification)
10. [Limitations](#10-limitations)
11. [Repository layout and usage](#11-repository-layout-and-usage)

---

## 1. Problem statement and constraints

The goal was to test **weight-space programming** ([blog post](https://rikiyat.github.io/blog/weight-space-programming/)) on a complete interactive program. The method has a general-purpose language model write a network's parameters directly from a natural-language specification, with no gradient-based optimization. The full specification is in [`GOAL.md`](GOAL.md). Its constraints were:

* **Parameters are text.** Every weight and bias is a float in a text file, and it may only be created or changed by editing that file.
* **No training.** No gradient descent, evolutionary search, RL or data fitting.
* **No generators.** No script, template or formula engine may emit parameter values.
* **Read-only debugging.** Forward passes may be run to observe failures. Interpreting the weights must be done by reading them directly, not with analysis tooling.
* **Thin host.** The framework only loads parameters, applies the fixed architecture, encodes input, supplies an RNG sample and renders output. It may not contain game logic.

Clarifications settled before implementation:

| Question | Resolution |
|---|---|
| Forward passes during debugging | Allowed |
| Randomness | The host supplies one uniform sample per step; mapping it to a piece is done in-network |
| I/O | The screen and key state go in; the next screen comes out |
| Board | Standard 10 × 20 |
| Size | Minimize, and justify every architectural choice |
| Activations | Unrestricted, with a stated rationale |

Acceptance criteria: a complete game is playable from start to top-out; all seven tetrominoes spawn, move, rotate, lock, clear and score correctly; a source audit finds no logic outside the network; and a design note documents every parameter file.

## 2. Related work

**Threshold logic.** McCulloch and Pitts (1943) showed that units computing a thresholded weighted sum realize Boolean functions, so networks of them can implement any finite-state machine. This project uses exactly that model: every node is a threshold gate over {0,1} inputs.

**Hand-set convolutional networks for cellular automata.** Springer and Kenyon ([2020](https://arxiv.org/abs/2009.01398)) hand-built a minimal convolutional network that implements Conway's Game of Life exactly. Their finding was that networks with the same minimal architecture rarely learned the rule through training, and training only succeeded reliably with substantial overparameterization. That is a direct argument for programming small exact networks rather than training them. Tetris goes well beyond Life, however. Life's update is a single local 3×3 rule. Tetris needs **global reductions** (is any cell of the candidate move blocked? is any row full? is the game over?), **conditional branching** on those reductions, **persistent counters** (gravity timer, score), **stochastic input** and **player input**. This work extends the hand-set convolutional approach with scalar reduction/broadcast nodes to cover those.

**Compiled networks.** Tracr ([Lindner et al., 2023](https://arxiv.org/abs/2301.05062)) compiles RASP programs into transformer weights to produce interpretability testbeds whose ground-truth mechanisms are known. Looped transformers ([Giannou et al., ICML 2023](https://proceedings.mlr.press/v202/giannou23a.html)) construct fixed weights so that a transformer run in a loop emulates a small instruction-set computer. Both show that weights can encode programs. Both, however, obtain the weights through a compiler or a mathematical construction. Here, each parameter is written individually, and the generator ban rules out a compiler. The looped transformers' execution model (the output re-enters as the next input) matches this project's screen-to-screen recurrence.

**Learned neural game engines.** GameNGen ([Valevski et al., 2024](https://arxiv.org/abs/2408.14837)) simulates DOOM in real time with a diffusion model trained on recorded agent play, alongside earlier learned simulators such as World Models and GameGAN. These systems trade exactness for scale and fidelity on far richer games, and their authors describe the result as approximate. This project takes the opposite point in that design space: a small game, but bit-exact state transitions with no drift over arbitrarily long play, and parameters that can be inspected one at a time.

**Positioning.** This is a hand-programmed, exact, weight-shared recurrent network that implements a complete interactive game. Its state is fully observable, and every parameter has a documented purpose.

## 3. System overview

<p align="center"><img src="docs/img/loop.svg" alt="Game loop" width="760"></p>

Formally, the network computes the transition function `s_{t+1} = f_θ(s_t, k_t, r_t)` of a discrete-time system, where:

* `s_t ∈ {0,1}^{12×21×16}` is the screen (12 binary channels on a 21 × 16 grid);
* `k_t ∈ {0,1}^4` is a one-hot (or all-zero) key vector (left, right, rotate, down);
* `r_t ~ U[0,1)` is the random sample.

The host calls `f_θ` at 10 Hz and feeds `s_{t+1}` back as the next input. The architecture has no hidden recurrent state: the screen *is* the state, so everything the game remembers is observable.

## 4. Computational model

### 4.1 Node types

The executor (`engine.js`) supports two node types. Both use the same activation.

* **Map node** (output in ℝ^{21×16}): `y = σ( Σ_i K_i ⋆ X_i + Σ_j w_j s_j + b )`, where `K_i ⋆ X_i` is a zero-padded, centered 2D cross-correlation of map input `X_i` with an `h_i × w_i` kernel, and each scalar input `s_j` is broadcast to all cells with weight `w_j`.
* **Scalar node** (output in ℝ): `y = σ( Σ_i w_i · ΣX_i + Σ_j w_j s_j + b )`, where map inputs are reduced by a global sum.

Map nodes provide translation-equivariant local rules. Scalar nodes provide global reductions (∃, counts) and gating signals, which are then broadcast back into map nodes. Together they express the global conditionals Tetris requires without any fully connected layers.

### 4.2 Activation

<p align="center"><img src="docs/img/gate.svg" alt="clamp01 and threshold logic" width="760"></p>

`σ(v) = clamp(v, 0, 1)`. All state is binary and all weights except the RNG thresholds are small integers, so every pre-activation is an integer. On integers, `σ` is an exact threshold gate:

| Function | Weights | Bias |
|---|---|---|
| AND of n inputs | 1, ..., 1 | −(n−1) |
| OR | 1, ..., 1 | 0 |
| NOT | −1 | 1 |
| at least k of n | 1, ..., 1 | −(k−1) |
| XOR(a, b) | `a + b − 2·AND(a,b)` | 0 (two layers) |

**Rationale.** ReLU would let an OR exceed 1 and need a renormalizing layer. Sigmoid and tanh never produce exact 0/1, so error would compound across steps of the recurrence. `clamp01` keeps the state space exactly `{0,1}` indefinitely. That is a fixed-point argument: the reachable state set is closed under `f_θ`, so there is no drift no matter how long the game runs.

### 4.3 Weight sharing

<p align="center"><img src="docs/img/stencil.svg" alt="Convolution as shift" width="760"></p>

Tetris rules are translation-invariant, so they are expressed as convolution kernels. A one-hot kernel is a shift operator: `[0 0 1]` gives `y(x) = X(x+1)`, a left shift. Shifts composed with pointwise logic give move, collision, rotation, spawn-copy and line-clear operations. Without weight sharing, each rule would need to be replicated across 336 cells, roughly 100 times more parameters. That would make manual authorship and inspection infeasible. Kernel window sizes are fixed in `graph.txt` (the architecture); the kernel values, including which offset is active, live entirely in the weight files.

## 5. State representation

<p align="center"><img src="docs/img/layers.svg" alt="The twelve channels" width="760"></p>

| Channel | Semantics | Rationale |
|---|---|---|
| `W` | walls and floor (static) | Collision becomes uniform overlap with `W ∨ L`; separating it from `L` keeps walls out of the row count. |
| `L` | locked cells | the stack |
| `A` | active piece | Transformed independently of `L`. |
| `P` | active pivot | Rotation center; empty for O. |
| `Q`, `R` | preview piece and pivot | Spawn reduces to a translation of `Q`, `R`. |
| `S` | score, 16-bit binary | |
| `T` | gravity timer token | |
| `G` | game-over flag (broadcast to all cells) | |
| `MQ`, `MS`, `MT` | static markers: preview anchor, score field, timer field | They let translation-equivariant kernels address specific screen regions. |

Grid layout: columns 0 and 11 are walls, columns 1 to 10 are the playfield and row 20 is the floor. Columns 12 to 15 hold the status area: the preview box in rows 0 to 1, the score in column 13 (rows 4 to 19, MSB on top) and the timer in column 15 (rows 4 to 10). Static channels are reproduced by identity nodes, so `f_θ` emits the complete next state.

## 6. Network walkthrough

<p align="center"><img src="docs/img/pipeline.svg" alt="Five stages" width="760"></p>

The graph is a DAG of 87 nodes evaluated in topological order. The notation below uses `∧ ∨ ¬` for the gates of §4.2, `shift_d(X)` for a one-hot kernel, `Σ` for a global sum and `[·]` for the clamp.

### 6.0 Global predicates

| Node | Definition | Meaning |
|---|---|---|
| `g0` | `[ΣG]` | the game is over |
| `aE` | `[1 − ΣA]` | no active piece |
| `F` | `[(1_{1×19} ⋆ L) − 9]` | cell lies in a full row |
| `fF` | `[ΣF]` | some row is full |
| `gt` | `[1 − ΣT]` | gravity tick (timer token has exited its field) |

`F`'s 19-wide all-ones kernel counts locked cells within ±9 columns. Only the 10 interior cells of a full row reach 10, and wall and status columns see at most 9. `g0` enters every later action gate with a negative weight, so after top-out the transition function becomes the identity on all gameplay channels.

### 6.1 Translation and collision

<p align="center"><img src="docs/img/move.gif" alt="Lateral movement against walls" width="260"></p>
<p align="center"><img src="docs/img/collision.svg" alt="Collision detection" width="760"></p>

Moves are evaluated speculatively:

* `ovL = shift_{+1}(A) ∧ (W ∨ L)`, the overlap map for a left move (`ovR` uses `shift_{−1}`)
* `okL = ¬∃ ovL`, computed as `[1 − Σ ovL]`
* `mL = kL ∧ okL ∧ ¬g0`, with weights (1, 1, −1) and bias −1

The same pattern recurs throughout: generate the candidate by shifting, intersect it with obstacles, reduce with ∃, gate on the result.

### 6.2 Rotation

<p align="center"><img src="docs/img/rotation.svg" alt="Rotation mechanism" width="760"></p>
<p align="center">
<img src="docs/img/rotate_T.gif" alt="T rotation" width="260">
<img src="docs/img/rotate_I.gif" alt="I rotation and refusal at wall" width="260">
</p>

Clockwise rotation in screen coordinates (y down) is `R(dx, dy) = (−dy, dx)`. For each offset `d` in the support set `D`, a detector node fires at cell `x` when the pivot is at `x − Rd` and a piece cell is at `x − Rd + d`:

```
U_d(x) = P(x − Rd) ∧ A(x − Rd + d)
```

Each `U_d` is an AND of two one-hot kernels (pivot at `−Rd`, piece at `d − Rd`) with bias −1. The result lands directly at the rotated position, so no second placement layer is needed.

| Nodes | Offsets `d` | Kernel windows | Used by |
|---|---|---|---|
| `U1` to `U4` | (±1,0), (0,±1) | 3×3, 3×3 | all but O |
| `U5` to `U8` | (±1,±1) | 3×3 (P), 5×5 (A) | S, Z, J, L, T |
| `U9` to `U12` | (±2,0), (0,±2) | 5×5, 5×5 | I only |

The pivots were chosen so that `|D| = 12` covers all pieces in all orientations. The per-node kernel coordinates are tabulated in `DESIGN.md`.

The rotated piece and its legality test:

* `AU = P ∨ ⋁_d U_d` (the pivot maps to itself)
* `ovU = AU ∧ (W ∨ L)`
* `okU = [ΣAU − 3 − 10·ΣovU]`, which is 1 iff the rotated piece has exactly 4 cells and no overlap

The cardinality term rejects rotations that lose cells across the top boundary, where zero padding drops them. It also rejects O without any special case: O carries no pivot, so `AU = ∅`.

Composition of the stage-1 result: `A1 = (shift_{+1}A ∧ mL) ∨ (shift_{−1}A ∧ mR) ∨ (AU ∧ mU) ∨ (A ∧ st1)`, with `st1 = ¬(mL ∨ mR ∨ mU)`. The gates are mutually exclusive because the host delivers at most one key per step. Each `∧ m` term is a separate node (`A1L`, `A1R`, `A1U`, `A1S`), because a single threshold cannot compute a sum of products. The pivot follows the same path through `P1L`, `P1R`, `P1S` and `P1`; rotation leaves `P` fixed.

### 6.3 Gravity and locking

<p align="center">
<img src="docs/img/drop_lock.gif" alt="Drop, lock and respawn" width="260">
<img src="docs/img/timer.gif" alt="Timer token" width="260">
</p>

**Timer.** The timer is a one-hot token that walks down a 7-cell field:

* `TS = shift_{down}(T) ∧ MT` advances the token; it drops out after the last cell.
* `TI = (MT ∧ ¬shift_{down}MT) ∧ gt` re-inserts it at the field's first cell. The kernel `[−1, 1, 0]ᵀ` detects that edge.

The period is 8 steps, or 0.8 s at 10 Hz. Timing is a game rule, so it is kept in-network rather than supplied by the host.

**Fall and lock.**

* `ovD = shift_{down}(A1) ∧ (W ∨ L)` and `okD = ¬∃ovD`
* `g = (gt ∨ kD) ∧ ¬g0`, as `[gt + kD − 2·g0]`
* `mD = g ∧ okD`, `lock = g ∧ ¬okD`, `st2 = ¬(mD ∨ lock)`
* `A2 = (shift_{down}A1 ∧ mD) ∨ (A1 ∧ st2)`; a locking piece leaves `A`...
* ...and enters `L` through `LK = A1 ∧ lock`.

### 6.4 Line clear

<p align="center"><img src="docs/img/rowfull.svg" alt="Row detection and clear" width="760"></p>
<p align="center"><img src="docs/img/line_clear.gif" alt="Four-line clear" width="260"></p>

* `C = [K_{39×1} ⋆ F]`, where the kernel is 1 for offsets 0..+19. It marks cells at or above a full row.
* `L' = (L ∧ ¬C) ∨ (shift_{down}L ∧ C) ∨ LK`

Each step clears one row: all cells above the lowest full row drop by one, and later full rows are handled on later steps. A fixed kernel cannot implement a data-dependent shift distance, and this sequencing avoids the need for one. It also makes the score increment exactly 1 per step that `fF` is set. Spawning is gated on `¬fF`, so no active piece coexists with a pending clear.

### 6.5 Score

<p align="center"><img src="docs/img/counter.svg" alt="Binary counter" width="760"></p>

The score is a 16-bit ripple incrementer, evaluated in parallel:

* `carry_i = fF ∧ MS_i ∧ ⋀_{j<i} S_j`. It is computed as `[fF + MS_i − 1 + Σ_{j<i}(S_j − MS_j)]`, where the sum over lower bits is 0 iff all of them are 1. A 31×1 kernel covers every lower bit.
* `SC = S ∧ carry`
* `S' = S + carry − 2·SC`, i.e. `S ⊕ carry`

### 6.6 Spawn and piece selection

<p align="center"><img src="docs/img/spawn.gif" alt="Spawn from preview" width="260"></p>
<p align="center"><img src="docs/img/random.svg" alt="RNG to one-hot" width="760"></p>

* **Spawn condition:** `e = aE ∧ ¬fF ∧ ¬g0`.
* **Spawn by translation:** `ASp = shift_{+8}(Q) ∧ e`, using a 1×17 kernel with its single 1 at the right edge (and the same for `PSp` on `R`). The preview box sits exactly 8 columns right of the spawn point.
* **Next state:** `Anew = A2 ∨ ASp` and `Pnew = P2 ∨ PSp`.
* **Thresholds:** `h_k = [7·10⁹ · r − k·10⁹]` approximates the step function `1[r ≥ k/7]`. The undefined band has width about 1.4·10⁻¹⁰, which is negligible.
* **One-hot selection:** `D_t = MQ ∧ h_t ∧ ¬h_{t+1} ∧ e`, with `h_0 ≡ 1` and `h_7 ≡ 0` folded into the biases. It lights exactly the anchor cell for exactly one piece, and only on spawn steps.
* **Preview render:** `Q' = (Q ∧ ¬e) ∨ Σ_t (K_t ⋆ D_t)`. `K_t` is the 5×5 shape of piece `t`, point-reflected: cross-correlation reads `X(x+k)`, so stamping a cell at offset `c` needs the weight at `−c`.
* **Preview pivot:** `R' = (R ∧ ¬e) ∨ ⋁_{t≠O} D_t`.

Spawn orientations and pivots (`P`):

```
I  .....      O  .XX.      T  .X.       S  .XX       Z  XX.       J  X..       L  ..X
   XPXX          .XX.         XPX          XP.          .PX          XPX          XPX
```

### 6.7 Game over

* `ovS = shift_{+8}(Q) ∧ L ∧ e`
* `ge = ∃ovS`
* `G' = g0 ∨ ge`, broadcast to every cell

The overlapping piece is still written into `A`, and the renderer shows the overlap in red. With `g0 = 1`, every action gate is zero and the dynamics freeze.

### 6.8 Static channels

`Wc`, `MQc`, `MSc` and `MTc` are identities (weight 1, bias 0), so the host never has to carry state forward.

**Network statistics:** 87 nodes and 1,035 parameters with 14 distinct values: {0, 1, −1, −2, −3, −9, −10, 7·10⁹, −1·10⁹, ..., −6·10⁹}. Most nodes have 2 to 6 parameters. The largest are the long reduction and shift kernels (`F` 19, `C` 39, `carry` 62, spawn copies 17 each) and `Qnew` (7 × 25). A per-file index is in [`DESIGN.md`](DESIGN.md).

## 7. Host framework

| File | Role |
|---|---|
| `graph.txt` | Architecture: node type, input wiring and kernel window sizes. It holds no parameter values. |
| `weights/<node>.txt` | Parameters: kernels in input order, then the bias. Pure numeric text. |
| `engine.js` | Generic executor (~150 LOC): cross-correlation, global sum, broadcast, clamp. It has no data-dependent branching. |
| `play.template.html` / `play.html` | Browser host: key queue (depth ≤ 2, one key per step), RNG sample, 10 Hz tick, per-channel rendering. `build.js` inlines all text files verbatim into a single self-contained `play.html`. |
| `sim.js`, `play.js` | Debug harnesses: scripted forward passes with seeded RNG and text rendering; `play.js` persists state between invocations so a game can be played one decision at a time. |
| `screen0.txt` | Initial state (walls and static markers), as a character map. |

[`AUDIT.md`](AUDIT.md) reviews every component and confirms the host contains no game logic. The renderer reads channels only to color them and never feeds anything back.

## 8. Design decisions

| Decision | Alternative considered | Reason |
|---|---|---|
| Observable screen-as-state | Hidden recurrent state | Required by the spec; also makes the system fully inspectable. |
| 12 binary channels | One categorical channel | Categorical decoding would cost extra layers; binary channels turn every rule into threshold logic. |
| Separate `W` channel | Treat padding as wall | Uniform collision test, and walls are excluded from the row count. |
| `clamp01` everywhere | ReLU / sigmoid / step | Exact gates on integer pre-activations; closed under recurrence (§4.2). |
| Convolutional weight sharing | Dense layers | ~100× fewer parameters; rules are written once. |
| Speculative shift + overlap test | Explicit legal-move computation | Minimal exact collision check that reuses one pattern. |
| Pivot-cell rotation with 12 detectors | Per-piece, per-orientation tables | One mechanism covers all pieces; O falls out of the cardinality check. |
| Classic rotation (I about a cell) | SRS with wall kicks | Kicks need many more nodes; this matches NES behavior. |
| Spawn by translating the preview | Render each spawn from scratch | One copy kernel. |
| One line per step | Simultaneous multi-line clear | Fixed kernels cannot do data-dependent shifts; exact line counting comes for free. |
| In-network gravity timer | Host-supplied tick | Timing is game logic. |
| Binary line counter | Classic point table | Multi-line clears are never observed atomically; binary fits 65,535 in 16 cells. |
| One key per step | Simultaneous keys | Mutually exclusive gates keep selection simple. |

## 9. Verification

All checks used forward passes with the outputs inspected visually. No tooling read the weights.

* **Movement and collision:** both walls block lateral moves; soft drop, locking and respawn behave correctly.
* **Rotation:** T and I cycle through all four orientations, and an I rotation into the right wall is refused.
* **Per-piece clear and score:** for each tetromino, `tests/clear_<piece>.txt` has a bottom row that is full except for that piece's footprint. In every case the piece moves, drops, completes the row, the row clears, the residual cells shift down and the score goes from 0 to 1.
* **Multi-line clear:** four rows clear on four consecutive steps, and the score counts in binary.
* **RNG coverage:** sampling `r` at each bin center produces I, O, T, S, Z, J and L.
* **End-to-end:** a complete game was played decision by decision with `play.js` (seed 42): 512 steps from an empty board to top-out, all seven pieces placed, one line cleared. The key trace is in `tests/full_game_seed42.txt`, and the final `play.html` replays it headless to the identical terminal state.

The parameters were correct on the first execution. The only defects found were in the host (a module export bug), in a test fixture (a piece drawn over a wall) and in one misreading of filtered output during the play-through, which single-step replay resolved. No parameter edits were needed.

## 10. Limitations

* Scoring counts lines rather than using the classic 100/300/500/800 point table.
* There is no hard drop, no hold and no lock delay.
* Rotation uses classic pivot rotation, not SRS with wall kicks.
* Pieces are drawn i.i.d. uniform; there is no 7-bag.
* Gravity speed is fixed; there are no levels.

Each of these is expressible in the same model by adding more nodes.

## 11. Repository layout and usage

```
README.md            this document
GOAL.md              original specification
DESIGN.md            design note with per-file parameter index
AUDIT.md             host framework audit
graph.txt            architecture
weights/             87 parameter files
screen0.txt          initial state
engine.js            executor
play.template.html   host page source
play.html            built, self-contained game
build.js             inlines sources into play.html
sim.js, play.js      forward-pass harnesses
tests/               fixtures and the recorded full game
docs/img/            figures and recorded animations
docs/make_gifs.py    renders animations from forward-pass output
```

```sh
node build.js                                                       # rebuild play.html after editing weights
node sim.js screen0.txt "..lllrrrdddd" 1 1                          # scripted run, text output
node sim.js screen0.txt "$(cat tests/full_game_seed42.txt)" 42 1    # replay the recorded game
python3 docs/make_gifs.py                                           # re-record animations (requires Pillow)
```

### References

* *Weight-Space Programming* (blog post). https://rikiyat.github.io/blog/weight-space-programming/
* W. S. McCulloch and W. Pitts, "A Logical Calculus of the Ideas Immanent in Nervous Activity," *Bulletin of Mathematical Biophysics*, 1943.
* J. M. Springer and G. T. Kenyon, "It's Hard for Neural Networks to Learn the Game of Life," 2020. https://arxiv.org/abs/2009.01398
* D. Lindner et al., "Tracr: Compiled Transformers as a Laboratory for Interpretability," 2023. https://arxiv.org/abs/2301.05062
* A. Giannou et al., "Looped Transformers as Programmable Computers," ICML 2023. https://proceedings.mlr.press/v202/giannou23a.html
* D. Valevski et al., "Diffusion Models Are Real-Time Game Engines," 2024. https://arxiv.org/abs/2408.14837
