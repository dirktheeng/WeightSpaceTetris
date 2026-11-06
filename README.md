# WeightSpaceTetris

**A complete, playable game of Tetris where every rule lives inside a small neural network whose numbers were written by hand.**

<p align="center"><img src="docs/img/full_game.gif" alt="A full game played on the network, from empty board to game over" width="300"></p>

Every frame above was produced by the network itself. Nothing was trained. No program generated the numbers. Each of the 1,035 numbers that make the game work was reasoned out and typed into a plain text file, and it can be read and explained one by one.

**Play it:** download the repository and open `play.html` in any browser. Use the arrow keys: left and right move, up rotates, down drops faster. P pauses and N starts a new game.

---

## Contents

1. [How this started: the goal](#1-how-this-started-the-goal)
2. [Where this fits among prior work](#2-where-this-fits-among-prior-work)
3. [The big picture](#3-the-big-picture)
4. [A neuron is a light switch with a dial](#4-a-neuron-is-a-light-switch-with-a-dial)
5. [The screen is the memory](#5-the-screen-is-the-memory)
6. [Stencils: one rule, written once, used everywhere](#6-stencils-one-rule-written-once-used-everywhere)
7. [One step of the game, stage by stage](#7-one-step-of-the-game-stage-by-stage)
8. [Every part of the network, explained](#8-every-part-of-the-network-explained)
9. [The code around the network](#9-the-code-around-the-network)
10. [The decisions, and why they were made](#10-the-decisions-and-why-they-were-made)
11. [How it was built, checked and debugged](#11-how-it-was-built-checked-and-debugged)
12. [Limits and honest caveats](#12-limits-and-honest-caveats)
13. [Files and how to run things](#13-files-and-how-to-run-things)

---

## 1. How this started: the goal

The project began with a challenge. Could a language model build Tetris using **weight-space programming**? That is the idea, described in [this blog post](https://rikiyat.github.io/blog/weight-space-programming/), that a model can write a neural network's weights directly from a description of what the network should do. Normally a network learns its weights from thousands of examples. Here, nothing is learned. The weights are *programmed*, the way you would write code.

The rules were strict, and they were agreed on before any work started. The full statement is in [`GOAL.md`](GOAL.md). In short:

* **Every weight and bias lives in a plain text file of numbers.** The only way to change one is to open the file and edit it.
* **No training of any kind.** No gradient descent, no evolution, no fitting to data.
* **No generators.** No script may produce weight values. Each number is decided by reasoning and written by hand.
* **Debugging is done by reading.** Running the game to watch a failure is allowed. Writing a program to analyze the weights is not; the cause has to be found by reading the numbers.
* **The framework around the network is thin.** It loads the files, passes in keys and a random number, and draws the screen. It may not contain any Tetris rule.

A few questions were settled in conversation before the goal was final:

| Question | Decision |
|---|---|
| May the game be run during debugging? | Yes, forward passes (running the network) are allowed. |
| Where does randomness come from? | The framework supplies one random number per step. The network turns it into a piece. |
| What goes in and what comes out? | The current **screen** plus keys go in, and the next **screen** comes out. |
| Board size? | Standard 10 wide by 20 tall. |
| How big should the network be? | As compact as possible, with every choice explainable. |
| Which activation functions? | Any, as long as the reason is stated. |

Success meant four things. A person can play a complete game. All seven pieces behave correctly. An audit shows no game logic outside the network. And a design note explains every part.

## 2. Where this fits among prior work

People have connected neural networks and logic, games and computation for a long time. This project borrows from several of those threads, and its particular combination is new.

**Neurons as logic gates (1943 onward).** The founding idea of the field was that an idealized neuron, adding up its inputs and comparing them to a threshold, behaves like a logic gate. With enough gates you can compute anything. This project rests directly on that idea: every one of its 87 nodes is a threshold gate doing AND, OR, NOT or "at least k".

**Hand-set weights for a cellular automaton.** The closest relative is work on Conway's Game of Life, a grid "game" whose rule looks only at a cell's neighbors. Springer and Kenyon pointed out that because Life's rule only looks at a 3x3 neighborhood, a purely convolutional network can compute it. They wrote such a network's weights by hand and got perfect accuracy. Their main finding is a strong argument for programming weights directly: networks with that same minimal shape, when *trained*, rarely found the solution, and training only succeeded reliably with much larger networks. Tetris is much harder than Life, though. Life's rule is purely local. Tetris needs global decisions: "is this *whole* move blocked?", "is this row full?", "has the game ended?". It also needs memory, randomness and player input. This project extends the hand-set convolutional approach to all of those.

**Compiling programs into network weights.** Tracr, from DeepMind, is a compiler that turns short human-readable programs into the weights of a transformer. It was built so interpretability researchers would have networks whose inner workings are known exactly. The looped transformers work went further: it set specific weights in a transformer and ran it in a loop so that it acted as a small general-purpose computer. Both show that networks can be *programmed* rather than trained. Both use a compiler or a mathematical construction to produce the weights. Here, no compiler exists. A model wrote each value directly and had to reason about it, which is exactly what the rules require. The looped transformers idea is close in spirit to the game loop used here, where the network's output is fed straight back in as its next input.

**Learned game engines.** At the other extreme, systems such as GameNGen learn to *imitate* a game from recorded play. GameNGen runs DOOM, a far richer game than Tetris, interactively and in real time, entirely inside a trained diffusion model. Its impressive visual quality comes from training on a large number of recorded play sessions, and its authors describe the result as close to the original but not an exact simulation. That approach trades exactness for scale and visual richness. This project makes the opposite trade: a small game, but every rule is exact, every value is visible, and the game never drifts, because every cell is always exactly 0 or 1.

**Weight-space programming.** The blog post that started this describes a language model writing working network weights from a spec, including small game-playing networks. This project applies that method to a complete, interactive game with state, randomness, scoring and a human player.

**In one sentence:** this is a *hand-written*, *exact*, *convolutional* network that runs an entire interactive game as a loop, screen in and screen out, with every number explained.

## 3. The big picture

<p align="center"><img src="docs/img/loop.svg" alt="The game loop: screen and inputs go into the network, the next screen comes out and is fed back" width="760"></p>

Ten times a second, the same thing happens:

1. The **current screen** goes into the network, along with which key (if any) was pressed and one random number.
2. The network computes the **next screen**.
3. The page draws that screen, then feeds it straight back in for the next step.

The network has no hidden memory. Everything the game must remember is on the screen: where the stack is, where the falling piece is, what the next piece will be, the score, a timer for gravity, and whether the game is over. If it isn't on the screen, the game has forgotten it.

## 4. A neuron is a light switch with a dial

<p align="center"><img src="docs/img/gate.svg" alt="clamp01 and the AND gate" width="760"></p>

Every node in this network does the same small job:

1. Multiply each input by a **weight** and add them up.
2. Add one more number, the **bias**.
3. **Clamp** the result between 0 and 1: anything below 0 becomes 0, anything above 1 becomes 1.

On the screen, every cell is either 0 (empty) or 1 (filled). With whole-number weights, that simple recipe turns each node into an exact logic gate:

* **AND** ("both are on"): weights 1 and 1, bias -1. The sum only reaches 1 when both inputs are 1.
* **OR** ("either is on"): weights 1 and 1, bias 0. The clamp stops the total at 1.
* **NOT** ("is off"): weight -1, bias 1.
* **"at least k of these"**: weights all 1, bias -(k-1).

**Why this activation?** The common alternatives fail in specific ways. A plain ReLU (zero below, unlimited above) would let an OR of two 1s become 2, which would need extra layers to tidy up. A smooth sigmoid never reaches exactly 0 or 1, so tiny errors would pile up over thousands of steps until the game broke. The clamp keeps every value exactly 0 or 1 forever, so the game is as exact on step 10,000 as on step 1.

## 5. The screen is the memory

<p align="center"><img src="docs/img/layers.svg" alt="The twelve layers of the screen" width="760"></p>

The screen is 21 rows by 16 columns. Think of it as **12 transparent sheets stacked on top of each other**, like the color layers of an image. Each sheet holds one kind of information, and every cell on every sheet is 0 or 1.

| Sheet | Holds | Why it is a separate sheet |
|---|---|---|
| **W** | walls and floor | Collision checks treat walls like solid blocks, so there are no special cases at the edges. Keeping them apart from the stack stops walls from counting toward a full row. |
| **L** | locked cells (the stack) | the pile you build |
| **A** | the falling piece | It must move separately from the stack. |
| **P** | the falling piece's pivot | Rotation needs a center point. |
| **Q** | the preview (next) piece | A new piece is a straight copy of this. |
| **R** | the preview piece's pivot | It is copied along with the preview. |
| **S** | score bits | a 16-digit binary number |
| **T** | the timer token | a ticking clock for gravity |
| **G** | game over | When set, the whole sheet lights up. |
| **MQ, MS, MT** | fixed markers | They tell the network *where* the preview box, score column and timer column are. |

The board is columns 0 to 11 (walls at 0 and 11, play area 1 to 10) and rows 0 to 19, with the floor on row 20. Columns 12 to 15 are the status area.

## 6. Stencils: one rule, written once, used everywhere

<p align="center"><img src="docs/img/stencil.svg" alt="A three-cell stencil sliding along a row shifts a piece left" width="760"></p>

A rule of Tetris works the same at every spot on the board: "move this piece one cell left" means the same thing everywhere. So instead of writing the rule 336 times, once per cell, it is written **once** as a small grid of numbers called a **stencil** (a convolution kernel), and the framework slides it across every cell.

The animation shows the simplest stencil, `[0 0 1]`. At each cell it looks at itself and its two neighbors. It ignores the left one and itself (weights 0) and copies the right one (weight 1). The effect is that the whole piece shifts one cell left. Change it to `[1 0 0]` and the piece shifts right. Make it tall instead of wide and the piece moves down.

**Why use stencils?** Without them, the same rule would have to be repeated at every cell. That would mean roughly 100 times more numbers: far too many to write, read or check by hand. With them, the whole game fits in 1,035 numbers, and most files hold between 2 and 6.

Some nodes need only a few global facts instead of a full sheet: "is anything blocked?", "is any row full?". Those are **scalar nodes**. A scalar node adds up an entire sheet into one number. Its single answer can then be fed back into every cell of a later sheet as an on/off switch.

## 7. One step of the game, stage by stage

<p align="center"><img src="docs/img/pipeline.svg" alt="The five stages of one network step" width="760"></p>

Each step runs top to bottom through 87 nodes in five stages. The next section opens up every one of them.

## 8. Every part of the network, explained

Node names below match the files in `weights/` and the lines in `graph.txt`.

### 8.0 Looking at the screen first

Before deciding anything, five nodes summarize the incoming screen:

| Node | Question it answers | How |
|---|---|---|
| `g0` | Is the game already over? | Adds up the G sheet; any lit cell means yes. |
| `aE` | Is there *no* falling piece? | NOT (sum of the A sheet). |
| `F` | Is this cell part of a full row? | A sideways stencil of 19 ones counts the stack 9 cells either side; bias -9 means "all 10". |
| `fF` | Is any row full? | Adds up F. |
| `gt` | Is it time for gravity? | True when the timer token has run off the end of its column. |

`g0` matters everywhere. Every later action subtracts it, so once the game is over, nothing can move.

### 8.1 Moving left and right, and running into things

<p align="center"><img src="docs/img/move.gif" alt="The piece moves left until it hits the wall, then right until it hits the other wall" width="260"></p>

<p align="center"><img src="docs/img/collision.svg" alt="How a collision is detected" width="760"></p>

The network never asks "can I move?" in the abstract. It **tries** the move and looks for trouble:

1. `ovL` builds the piece shifted one cell left, using the stencil `[0 0 1]`. It ANDs each cell with "wall OR stack" (bias -1). Any lit cell in `ovL` is an overlap.
2. `okL` sums `ovL` over the whole board: **ok to move left = NOT (any overlap)**.
3. `mL` = left key AND `okL` AND NOT game over. That is three conditions, so the weights are 1, 1 and -1 with bias -1.
4. `ovR`, `okR` and `mR` do the same for moving right.

In the animation the piece slides to the left wall and stops: the next left move would overlap the wall, so `okL` becomes 0. Then it slides to the right wall and stops there too.

### 8.2 Rotation

<p align="center"><img src="docs/img/rotation.svg" alt="How rotation works" width="760"></p>

<p align="center">
<img src="docs/img/rotate_T.gif" alt="T piece rotating through all four positions" width="260">
<img src="docs/img/rotate_I.gif" alt="I piece rotating, and a rotation refused at the wall" width="260">
</p>

Rotation is the cleverest part. Each piece has a **pivot** cell, marked on the P sheet, that stays put while the other cells swing a quarter turn clockwise around it. On the screen (where down is positive), "one to the right" becomes "one below", "one below" becomes "one to the left", and so on.

Twelve detector nodes, `U1` to `U12`, each handle one possible offset from the pivot. Each one asks a two-part question at every cell: *"Is the pivot where it would need to be, AND is there a piece cell at the spot that would swing onto me?"* That is an AND of two one-hot stencils with bias -1. If the answer is yes, this cell belongs to the turned piece.

| Detectors | Offsets covered | Who needs them |
|---|---|---|
| `U1`..`U4` | directly beside the pivot (right, below, left, above) | every piece except O |
| `U5`..`U8` | diagonal from the pivot | S, Z, J, L, and T when turned |
| `U9`..`U12` | two cells away in a straight line | only the long I piece |

`AU` combines the pivot cell with all twelve detectors to form the turned piece. Then the same "try it and check" idea applies. `ovU` looks for overlaps, and `okU` allows the turn only if the turned piece **still has all 4 cells AND overlaps nothing**. The "4 cells" test refuses a turn that would push a cell off the top of the screen. It also handles the O piece with no special code: O has no pivot, so its turned version is empty and is always refused.

In the I animation, the last turn is refused because the piece is against the right wall and turning it would push cells into the wall.

**Combining the player's move:** `A1L`, `A1R`, `A1U` and `A1S` each switch one candidate piece on or off (shifted left, shifted right, turned, or unchanged), and `A1` ORs them. Exactly one switch is on, because the framework sends at most one key per step and `st1` ("no move happened") is 1 only when no move was made. The pivot follows the same path through `P1L`, `P1R`, `P1S` and `P1`. A rotation leaves the pivot where it is.

### 8.3 Gravity, dropping and locking

<p align="center">
<img src="docs/img/drop_lock.gif" alt="A piece drops, locks into the stack, and the next piece spawns" width="260">
<img src="docs/img/timer.gif" alt="The timer token ticking down its column" width="260">
</p>

**The clock.** Pieces fall on their own every 0.8 seconds. Since the network has no hidden memory, the clock has to be on the screen: a single green **token** walks down a 7-cell column, one cell per step (right-hand animation).

* `TS` moves the token down one cell and keeps it inside its column using the MT marker, so it falls off after the last cell.
* When the column is empty, `gt` says "gravity now". `TI` then places a new token at the top of the column. It finds the top by looking for "column here AND no column above".
* `Tnew` combines the two. One full cycle is 8 steps, which is 0.8 seconds at 10 steps per second.

**Falling or locking.**

* `ovD` tries the piece one row lower (a tall stencil reading the cell above) and checks it against "wall OR stack". `okD` = NOT any overlap.
* `g` = (gravity tick OR down key) AND NOT game over. The weight on game over is -2, so it cancels even when both inputs are on.
* `mD` = g AND okD: fall one row.
* `lock` = g AND NOT okD: the piece can't go lower, so it locks into the stack.
* `st2` = neither one happened.
* `A2D`, `A2S`, `P2D` and `P2S` produce the fallen or unchanged piece and pivot. A locking piece goes into neither, so it disappears from the falling sheet...
* ...and reappears in the stack: `LK` = piece AND lock adds its cells to L.

### 8.4 Clearing full lines

<p align="center"><img src="docs/img/rowfull.svg" alt="How full rows are found and cleared" width="760"></p>

<p align="center"><img src="docs/img/line_clear.gif" alt="Four lines clear one per step while the score counts up" width="260"></p>

* `F` (from 8.0) lights every cell of a full row.
* `C` uses a tall stencil (20 ones looking downward) to mark every cell that has a full row at or below it. Those are the cells that must move down.
* `LKeep` = stack AND NOT marked: these cells stay where they are.
* `LSh` = the cell above AND marked: these cells take on whatever was directly above them.
* `Lnew` = `LKeep` OR `LSh` OR `LK`.

**One row per step.** If several rows are full, everything above the lowest one drops by one row, and on the next step the next full row is handled. The animation shows four lines clearing on four consecutive steps. This keeps the network small, because a single stencil can't shift things by a variable amount. It also makes scoring exact, since exactly one line is cleared per step. To stay safe, no new piece appears until every full row is gone.

### 8.5 The score

<p align="center"><img src="docs/img/counter.svg" alt="The binary score counter" width="760"></p>

The score counts cleared lines in **binary**, shown as the column of 16 cells next to the board (top is the high digit, bottom is the low digit). Adding 1 to a binary number means "flip digits from the bottom up until you flip a 0 to 1". Three nodes do that:

* `carry`: a digit flips if a line is being cleared (`fF`) AND every digit below it is already 1. A tall stencil checks all the lower digits at once. Each one contributes "digit minus field", which is 0 when the digit is 1 and -1 when it is 0, so the sum is 0 only when all of them are 1.
* `SC` = digit AND carry.
* `Snew` = digit + carry - 2 x `SC`. That is "exclusive or": the result is 1 when exactly one of them is 1.

### 8.6 New pieces: the preview and the random choice

<p align="center"><img src="docs/img/spawn.gif" alt="The first preview is drawn, then copied into the board while a new preview is chosen" width="260"></p>

<p align="center"><img src="docs/img/random.svg" alt="Turning a random number into one of seven pieces" width="760"></p>

**When to spawn.** `e` = no falling piece AND no full row AND NOT game over.

**Spawning is a copy.** The preview box sits 8 columns to the right of the spawn spot. `ASp` and `PSp` use a 17-wide stencil whose only 1 sits at the far right, so it reads 8 columns over. That copies the preview piece and its pivot into the board. `Anew` and `Pnew` then combine the new piece with whatever happened in the earlier stages.

**Choosing the next preview.**

* `h1`..`h6` are threshold nodes: "is the random number at least 1/7? 2/7? ... 6/7?". They use a very steep slope (weight 7,000,000,000) so their answer is always a clean 0 or 1.
* `D0`..`D6`: piece *k* is chosen when threshold *k* is yes and threshold *k+1* is no. Each `D` node also requires "a spawn is happening" AND "this is the preview anchor cell", so it lights exactly one cell, and only at the right moment.
* `Qnew` stamps the chosen piece's shape into the preview box using a 5x5 stencil per piece. The stencils are mirror images of the shapes, because a stencil *reads* from offsets. Copying a cell to an offset means putting the 1 at the opposite offset.
* `QK` and `RK` keep the old preview when nothing spawns. `Rnew` places the preview pivot for every piece except O.

**The seven pieces and their pivots** (pivot marked P):

```
I  .....      O  .XX.      T  .X.       S  .XX       Z  XX.       J  X..       L  ..X
   XPXX          .XX.         XPX          XP.          .PX          XPX          XPX
```

The pivots were chosen so that every cell of every rotation stays within two steps of the pivot. That keeps the number of rotation detectors at exactly 12.

### 8.7 Game over

<p align="center"><img src="docs/img/game_over.gif" alt="The stack reaches the top; the new piece overlaps and the game freezes" width="260"></p>

* `ovS` checks whether the piece about to be copied in overlaps the stack, but only on a spawn step (bias -2 for a three-way AND).
* `ge` = any overlap. `Gnew` = old game over OR new game over, lit across the whole sheet.

The overlapping piece is still drawn, in red, so the player sees what topped out. From then on `g0` is 1, which switches off every move, every fall and every spawn, and the game freezes.

### 8.8 Keeping the fixtures

`Wc`, `MQc`, `MSc` and `MTc` simply copy the walls and the three markers (weight 1, bias 0). This way the network, not the framework, produces the *entire* next screen.

## 9. The code around the network

The surrounding code is deliberately small and contains no Tetris rules. [`AUDIT.md`](AUDIT.md) goes through it piece by piece.

| File | What it does |
|---|---|
| `graph.txt` | The wiring diagram: for each node, its type, its inputs and the size of each stencil. It holds no numbers that decide behavior. |
| `weights/*.txt` | One file per node: the stencil values in input order, then the bias. Only numbers. |
| `engine.js` | The generic executor, about 150 lines. It knows only "slide a stencil", "add up a sheet", "add a switch value to every cell" and "clamp". It never looks at what a cell means. |
| `play.template.html` → `play.html` | The browser page. Every 100 ms it takes one key from a short queue, adds a random number, runs the network and paints the sheets in colors. `build.js` pastes the engine and every text file into `play.html` unchanged, so the game is one self-contained file. |
| `sim.js`, `play.js` | Debugging tools. They run the network on a list of keys and print the screens as text, so behavior can be watched step by step. |
| `screen0.txt` | The starting screen, drawn with characters: walls, floor and the three markers. |

## 10. The decisions, and why they were made

| Decision | Alternatives | Why this one |
|---|---|---|
| Screen in, screen out, with no hidden memory | Keep a hidden state vector | This was the agreed rule. It also makes the network fully inspectable: everything it "knows" is visible. |
| 12 binary sheets | One sheet with codes (0 empty, 1 locked, 2 falling...) | Decoding codes would need extra layers. Separate on/off sheets make every rule a simple AND/OR. |
| Walls as their own sheet | Treat the screen edge as a wall | No special edge cases in collisions, and walls never count toward full rows. |
| One activation, clamp01 | ReLU, sigmoid, step | Exact logic on whole numbers; values stay exactly 0 or 1 forever (section 4). |
| Stencils (weight sharing) | Fully connected layers | One rule is written once instead of 336 times, which made writing and reading by hand possible. |
| "Try the move, then check for overlap" | Compute legal moves directly | A shifted copy plus one overlap test is the simplest exact collision check. |
| Rotation about a pivot cell | Lookup tables for each piece and orientation | 12 small detectors handle all seven pieces in all four orientations. |
| O piece has no pivot | Special-case code for O | The general "must still have 4 cells" rule refuses it for free. |
| I rotates about one of its own cells | Modern rotation rules with wall kicks | Kicks would need many more nodes. Classic NES Tetris also turns the I about a cell. |
| Spawn by copying the preview | Draw each new piece from scratch | The preview already shows the shape, so one copy stencil does the job. |
| One line cleared per step | Clear all full rows at once | A fixed stencil can't shift by a variable amount. This also makes the line count exact. |
| Gravity clock as a token on the screen | Have the framework send a "gravity" signal | Timing is a game rule, so it belongs in the network. |
| Score counts lines, in binary | Classic points (100/300/500/800) | With one line per step the network never sees a "4-line clear" at once. Binary fits 65,535 lines in 16 cells. |
| At most one key per step | Allow simultaneous keys | No conflicts between moves, so the selection logic stays simple. |

## 11. How it was built, checked and debugged

**Building.** The design came first: the sheets, the stages and the logic for each node, worked out in reasoning. Then every weight file was typed by hand, one at a time. Only 14 distinct values appear in the entire network: 0, 1, -1, -2, -3, -9, -10, and the seven numbers in the random-number thresholds (7,000,000,000 and -1,000,000,000 through -6,000,000,000).

**Checking.** The game was run with `sim.js` and the printed screens were read by eye:

* moving into both walls stops the piece; dropping, locking and new pieces work;
* T and I rotate through all four positions, and a rotation into a wall is refused;
* for **each** of the seven pieces, a hand-drawn test screen has a bottom row full except for exactly that piece's shape. Each piece was moved, dropped, filled the gap, cleared the line and raised the score from 0 to 1 (`tests/clear_*.txt`);
* four lines clear on four consecutive steps, with the score counting in binary;
* sweeping the random number across all seven ranges produces I, O, T, S, Z, J and L;
* a **complete game** was played one decision at a time with `play.js`, choosing each move from the printed screen. It ran 512 steps, from an empty board to game over, used all seven pieces and cleared a line. The full key record is `tests/full_game_seed42.txt`, and the final `play.html` replays it to the same ending.

**Debugging.** The weights behaved as designed on the first run. The bugs that appeared were outside the network. The engine's export line didn't work in Node (a framework fix). A test screen had a piece accidentally drawn over a wall (a test-file fix). Once, during the play-through, a text filter hid the rows where a piece was falling, which made the network look as if it had spawned the wrong piece. Stepping through one tick at a time showed the network was right and the reading was wrong. No weight needed changing. Had one been wrong, the rule was clear: find the cause by reading the numbers, and fix it by editing the file.

## 12. Limits and honest caveats

* **Scoring** counts lines, not the classic point values.
* **No hard drop** and no "hold piece". Only soft drop is wired up.
* **Rotation** follows the classic style, not the modern rotation system with wall kicks.
* **The piece order** is fully random, with no "bag" that guarantees all seven appear in every set of seven.
* **Speed** stays constant; there are no levels.
* Every one of these could be added the same way: by writing more nodes and weights by hand.

## 13. Files and how to run things

```
README.md            this explainer
GOAL.md              the original goal statement
DESIGN.md            technical design note, with a table for every weight file
AUDIT.md             line-by-line audit of the framework code
graph.txt            network wiring (no values)
weights/             87 hand-written weight files (1,035 numbers)
screen0.txt          starting screen
engine.js            generic network executor
play.template.html   browser page source; build.js turns it into play.html
play.html            the playable game (open this)
sim.js, play.js      forward-pass debugging tools
tests/               test screens and the recorded full game
docs/img/            the diagrams and animations in this README
docs/make_gifs.py    records the animations from real network output
```

* **Play:** open `play.html`.
* **Rebuild the page after editing a weight file:** `node build.js`
* **Watch a scripted run as text:** `node sim.js screen0.txt "..lllrrrdddd" 1 1`
* **Replay the recorded game:** `node sim.js screen0.txt "$(cat tests/full_game_seed42.txt)" 42 1`
* **Re-record the animations:** `python3 docs/make_gifs.py` (needs Pillow). Every frame is a screen the network produced; the script only paints them.

### References

* Weight-space programming: https://rikiyat.github.io/blog/weight-space-programming/
* Springer and Kenyon, *It's Hard for Neural Networks to Learn the Game of Life* (2020): https://arxiv.org/abs/2009.01398
* Lindner et al., *Tracr: Compiled Transformers as a Laboratory for Interpretability* (2023): https://arxiv.org/abs/2301.05062
* Giannou et al., *Looped Transformers as Programmable Computers* (ICML 2023): https://proceedings.mlr.press/v202/giannou23a.html
* Valevski et al., *Diffusion Models Are Real-Time Game Engines* (GameNGen, 2024): https://arxiv.org/abs/2408.14837
* McCulloch and Pitts, *A Logical Calculus of the Ideas Immanent in Nervous Activity* (1943), the origin of the neuron-as-logic-gate idea.
