# daw

A mathematical and programmatic audio engine. Signals are
[SymPy](https://www.sympy.org) expressions of a single shared time symbol,
`daw.waves.time.TIME`. Oscillators, envelopes, LFOs and time shifts are all
plain algebra, and an expression is only turned into samples when you play,
save or plot it.

```python
from daw.waves.basic import sine
from daw.waves.utils import play

play(sine(440), duration=2.0)
```

## How it works

1. Build an expression: `sine(440) * 0.5`, `square(lfo)`, `adsr_envelope(...)`.
2. `compile(expr, duration, sample_rate)` runs `sympy.lambdify` over a NumPy
   time axis and returns a sample array.
3. `play`, `save` (WAV via `soundfile`) and `plot` call `compile` for you.

Frequencies may themselves be expressions (e.g. `sine(440 + 5 * sine(6))`).
The phase is obtained by integrating the frequency over `TIME`, so frequency
modulation stays correct.

## Layout

| Path | Contents |
| --- | --- |
| `src/daw/constants.py` | `Note` frequency table, default sample rate |
| `src/daw/waves/basic.py` | `sine`, `square`, `triangle`, `sawtooth`, `trigger` |
| `src/daw/waves/time.py` | `TIME` symbol and the `BPM` helper |
| `src/daw/waves/utils.py` | `compile`, `play`, `save`, `plot`, `WHITE_NOISE` |
| `src/daw/envelope.py` | `adsr_envelope` and the `ACTIVE_DURATION` symbol |
| `src/daw/filters.py` | First-order low-pass filter, applied on the sample grid |
| `examples/` | Example scripts (`kick.py`, `ode_to_joy.py`, ...) |
| `tests/` | Pytest suite, golden files and guard tests |
| `tools/` | `mutation_check.py`, which proves the tests catch bugs |
| `guidelines.md` | How we test, lint and commit. Read before contributing |

## Development

The Nix flake provides a dev shell with Python, the project dependencies and
the tooling below:

```sh
nix develop
```

or run a single command without entering the shell, e.g.
`nix develop -c pytest`.

Without Nix, install the package in editable mode with the dev extras:

```sh
pip install -e '.[develop]'
```

| Task | Command |
| --- | --- |
| Run tests | `pytest` (add `-m "not slow"` to skip the examples) |
| Lint | `ruff check .` |
| Format | `ruff format` |
| Type check | `mypy src/daw` |

Tool settings live in `pyproject.toml` (ruff, mypy, pytest).

`nix flake check` runs the tests, ruff and mypy in the Nix sandbox.

Every change, new feature and fixed bug needs rigorous tests. The standard,
per-feature checklists and conventions are in [guidelines.md](guidelines.md).

## Running the examples

```sh
python examples/kick.py
python examples/ode_to_joy.py
```

The dev shell puts `src/` on `PYTHONPATH`. Outside it, install the package with
`pip install -e .` first.

Playback needs [PortAudio](https://www.portaudio.com) via `sounddevice`.
`sounddevice` is only imported when `play` is called, so compiling, saving and
the test suite work on machines without an audio backend.

## Known limitations

- A filtered signal can only be evaluated over a uniform grid of sample
  times (which is what `compile`, `play` and `save` use), not at a single
  time, and the filter starts at rest at the first sample.
- White noise is a single shared symbol, so every use of `WHITE_NOISE` in one
  expression sees the same samples.
