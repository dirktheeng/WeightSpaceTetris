# Goal Statement: Tetris by Weight-Space Programming

## Objective
Build a playable, standard 10×20 game of Tetris in which all of the game logic lives in the weights and biases of a hand-written artificial neural network. The network computes piece movement, rotation, collision, locking, line clearing, scoring, piece spawning and game over. A thin serving framework only carries data in and out of it. This follows the weight-space programming approach (https://rikiyat.github.io/blog/weight-space-programming/), where a general-purpose LLM reads and writes a network's weights directly from a natural-language spec, without an optimizer or gradient descent.

## The network as a screen-to-screen step function
At each step the network gets three inputs: the current screen, the keyboard state and a random number from the framework. Its output is the next screen. The screen is the only game state, so everything the game needs to remember has to be visible on it or encoded in it. That includes locked cells, the falling piece (with a cell value distinct from locked cells), the next piece, the score and the game-over condition. The framework draws the screen and passes it back in unchanged on the next step.

## Rules for the weights
1. Every weight and bias is stored as plain floating point values in text files. The only way to create or change them is to read and write those text files directly.
2. No training of any kind. That means no gradient descent, no evolutionary search, no fitting to data, and no reinforcement learning.
3. No generators. No program, script, formula engine or template may produce weight values. Each value is decided by reasoning and written by hand into the file.
4. Interpretation and debugging happen by reading the weight files directly. No program may analyze, summarize, visualize or explain the weights. Forward passes are allowed during debugging, so the game can be stepped or played to watch a failure. The cause is found by reading the values, and the fix is made by editing them.

## Design principles
1. **Compactness.** The network should be as small as possible: few layers, few units, few distinct values.
2. **Explainability.** Every architectural choice can be justified in plain language. That covers each layer, each activation function, and how pieces and cells are represented. Every group of weights has a stated purpose.
3. **Shared structure.** Tetris rules are the same at every board position. So the network may use weight-sharing layers, such as small convolution-style kernels applied across the board. The framework applies each kernel, and each kernel is written once by hand in its own file. This keeps the network compact and readable, and it isn't a generator because the framework never makes values. It only applies the hand-written ones at each position. The sharing pattern is part of the fixed architecture and has to be explained like any other choice.
4. **Activations.** Any activation function may be used (step, ReLU, clamp, or others) as long as the reason it was chosen is stated.

## What the framework may do
The framework loads the weight files and applies the fixed architecture, including the weight-sharing patterns. It turns keyboard input into input values, supplies the random number, renders the output screen and feeds it back in. It contains no Tetris logic. It must not check collisions, clear lines, choose pieces, track score or correct the network's output in any way.

## Success criteria
1. A person can play a complete game on a standard 10×20 board from start to game over.
2. All seven tetrominoes appear, and they spawn, move, rotate, fall, lock, clear lines and score correctly.
3. An audit of the framework source shows no game logic outside the network.
4. A written design note explains the architecture, the representation of cells and pieces, and the purpose of every weight file.
