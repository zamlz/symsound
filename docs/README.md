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
| [Combining signals](combining.md) | Mixing, gating, fading, crossfading, the building blocks of `kick.py` |
| [Playing, saving and plotting](output.md) | `compile`, `play`, `save`, `plot` |
| [`kick.py` walkthrough](kick-walkthrough.md) | The whole example explained line by line |
| [Gotchas](gotchas.md) | Limits and surprises worth knowing before they bite |

Every `python` code block in these pages is run by the test suite
(`tests/test_docs.py`), so the examples stay correct. The blocks in a page share
one namespace and run top to bottom, so later blocks may use names defined by
earlier ones on the same page.
