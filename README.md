# WeightSpaceTetris

This is a playable, standard 10×20 Tetris in which every game rule lives in the hand-written weights and biases of a small neural network. It follows the weight-space programming approach (https://rikiyat.github.io/blog/weight-space-programming/): nothing was trained or generated, and every value was reasoned out and typed into a text file.

* **Play:** open `play.html` in a browser. Use the arrow keys to move, rotate and soft drop, P to pause and N for a new game.
* **How it works:** see `DESIGN.md`, which explains the architecture, the screen layout and every weight file.
* **Framework audit:** see `AUDIT.md`.
* **Goal statement:** see `GOAL.md`.

## Layout

```
graph.txt            architecture (wiring and kernel windows, no values)
weights/*.txt        87 hand-written weight files, 1,035 numbers
screen0.txt          starting screen (walls and fixtures)
engine.js            generic executor: conv, global sum, broadcast, clamp01
play.template.html   browser front end; build.js inlines files into play.html
sim.js, play.js      forward-pass debug harnesses
tests/               hand-drawn test screens and a full recorded game
```

Rebuild the page after editing weights: `node build.js`.
Replay the recorded game: `node sim.js screen0.txt "$(cat tests/full_game_seed42.txt)" 42 1`.
