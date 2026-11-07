# daw documentation

These pages explain how to use the pieces of the library, not how they are
implemented. Read them in order the first time, then use them as reference.

| Page | What it covers |
| --- | --- |
| [Concepts](concepts.md) | What a signal is, `TIME`, and how an expression becomes sound |
| [Oscillators](oscillators.md) | `sine`, `square`, `triangle`, `sawtooth`, notes, frequency modulation |
| [Time and rhythm](time-and-rhythm.md) | `BPM`, shifting a signal in time, placing notes in a song |
| [Envelopes](envelopes.md) | `adsr_envelope` and shaping a sound's loudness over time |
| [Filters](filters.md) | `low_pass_filter`, including sweeping the cutoff |
| [Delay, echo and reverb](effects.md) | `delay`, and how to build an echo and a reverb from it |
| [Combining signals](combining.md) | Mixing, gating, fading, crossfading, the building blocks of `house_with_low_pass.py` |
| [Playing, saving and plotting](output.md) | `compile`, `play`, `save`, `plot` |
| [Visualizing a signal](visualizing.md) | `visualize`, `snapshot`: a window with a scope, a spectrogram and a seek bar; `Player` |
| [A playable keyboard](keyboard.md) | `keyboard`, `keyboard_snapshot`: play a signal with your computer's keys; `LiveSpectrogram` |
| [Live playback](live.md) | `NoteSource`, `Mixer`, `Instrument`, `RingBuffer`, `LiveOutput`: notes rendered in real time |
| [`house_with_low_pass.py` walkthrough](house-with-low-pass-walkthrough.md) | The whole example explained line by line |
| [Gotchas](gotchas.md) | Limits and surprises worth knowing before they bite |

The code blocks in these pages are real. They are run by
[mdutils](https://github.com/zamlz/mdutils) (`md code`), which writes what they
print under them, and the graphs are drawn by the blocks themselves with `plot`
and `spectrogram`. Each block runs on its own in a fresh Python, so every block
starts with its imports and you can copy any one of them into a file and run it.
`nix flake check` re-runs them all and fails if the committed output or graphs
differ. To regenerate after a change: `nix develop -c python tools/build_docs.py`.
