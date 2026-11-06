# Framework Audit

This audit covers every piece of code outside the weight files. For each one, it asks whether the code holds any Tetris rule: collision, movement, rotation, gravity, locking, line clearing, piece choice, spawning, scoring or game over.

## engine.js (the executor)

| Part | What it does | Game logic? |
|---|---|---|
| `H = 21, W = 16`, `CHANNELS`, `SCALAR_INPUTS` | Screen size and the names of its 12 planes and 5 scalar inputs. | No. These are the input and output shape, like the size of an image. |
| `LEGEND` | Maps characters in screen files to channel bits, for writing start screens and printing screens. | No. It's an encoding table only. |
| `acts` | `clamp01` and `id`. | No. These are generic activations. |
| `parseGraph` | Reads node lines: type, name, activation, inputs and window sizes. | No. It's a parser. |
| `parseNums`, `attachWeights` | Reads the numbers from `weights/NAME.txt` and splits them into kernels and a bias by input order. It checks that the count matches. | No. It never creates, changes or rounds a value. |
| `conv` | Zero-padded, centered 2D convolution: `acc[y,x] += k[i,j] * in[y+i-ch, x+j-cw]`. | No. It's the same operation for every node. |
| `total` | Sum of a map. | No. It's the generic global-sum used by scalar nodes. |
| `step` | Evaluates the nodes in file order. A map node is bias plus convolutions plus broadcast scalars, then the activation. A scalar node is bias plus weighted sums or values, then the activation. The next screen is copied from the nodes named in `out` lines. | No. It contains no branches on screen content, and no names of game concepts other than the output mapping that `graph.txt` gives. |
| `decodeScreen`, `encodeScreen` | Turn screen text into channels and back. `encodeScreen` prints "GAME OVER" when the network's `G` plane is lit. | No. They are display and I/O only. |

`engine.js` has no condition on any cell value. Its only conditionals are loop bounds, zero padding at the screen edge, skipping zero kernel entries (a speed shortcut with the same result) and error checks.

## play.template.html (the browser page)

| Part | What it does | Game logic? |
|---|---|---|
| Key handler | Turns the arrow keys into `l r u d` and queues at most 2 of them. P pauses and N restarts from `screen0.txt`. | No. It's input capture. The network decides whether a key has any effect. |
| `tick` | Every 100 ms: takes one queued key, builds `kL kR kU kD` and `rng = Math.random()`, calls `WSP.step`, stores the returned screen and draws it. | No. The returned screen is used unchanged. |
| `draw` | Paints each lit plane in a color. If the `G` plane is lit, it tints the board and shows a message. | No. It's rendering only. Nothing it reads is fed back. An earlier version also turned the score cells into a decimal number for display. That code was removed so the page only renders. |

## build.js

It reads `engine.js`, `graph.txt`, `screen0.txt` and every `weights/*.txt`, and pastes them into `play.html` verbatim as text. It doesn't parse, compute or alter any value. It's packaging only.

## sim.js and play.js (debug harnesses)

They load the same files, run `WSP.step` on a key script with a seeded random number, and print the returned screens. `play.js` saves the screen between calls so a game can be played one decision at a time. Neither inspects weights or changes a screen.

## screen0.txt

This is the starting screen. It holds the walls and floor, the preview anchor, the score field and the timer field. It's the initial state, the way a cartridge shows an empty board; it doesn't define any rule. Every later screen is produced by the network.

## Result

There is no Tetris rule anywhere outside `graph.txt` and `weights/`. `graph.txt` itself holds only wiring and window sizes, and every number that decides behavior is in a weight file.
