# daw

$$
\begin{aligned}
d(t) ={}& 0.8\Bigl( \left(\left\lfloor \tfrac{7t}{3} + 1 \right\rfloor - \tfrac{7t}{3}\right)^{1/2} \sin(80\pi t) \\
&+ 0.25 \left(\left\lfloor \tfrac{7t}{3} + \tfrac{3}{2} \right\rfloor - \tfrac{7t}{3} - \tfrac{1}{2}\right)^{20} n(t) \\
&+ 0.2 \left(\left\lfloor \tfrac{7t}{12} + \tfrac{3}{2} \right\rfloor - \tfrac{7t}{12} - \tfrac{1}{2}\right)^{9}
\bigl(1000t - \lfloor 1000t + \tfrac{1}{2} \rfloor\bigr)
\left(\mathrm{sgn}\sin\!\Bigl(2\pi\bigl(12.5\,t - \tfrac{36}{7\pi}\cos\tfrac{7\pi t}{6}\bigr)\Bigr) + 1\right)\Bigr)
\end{aligned}
$$

> Equation from `./examples/house.py`,
> where $n(t) \sim U(-1, 1)$ is white noise, a new random value at every sample.

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

## Documentation

See [docs/](docs/README.md) for how to use each piece, with graphs of the
signals and a line-by-line walkthrough of `examples/house.py`. The code blocks
in the docs are run by [mdutils](https://github.com/zamlz/mdutils) (`md`,
provided by the dev shell), which also draws the graphs:

```sh
nix develop -c python tools/build_docs.py          # regenerate output and graphs
nix develop -c python tools/build_docs.py --check  # fail if they are stale
```

## Layout

| Path | Contents |
| --- | --- |
| `src/daw/constants.py` | `Note` frequency table, default sample rate |
| `src/daw/waves/basic.py` | `sine`, `square`, `triangle`, `sawtooth`, `trigger` |
| `src/daw/waves/time.py` | `TIME` symbol and the `BPM` helper |
| `src/daw/waves/utils.py` | `compile`, `play`, `save`, `plot`, `WHITE_NOISE` |
| `src/daw/envelope.py` | `adsr_envelope` and the `ACTIVE_DURATION` symbol |
| `src/daw/filters.py` | First-order low-pass filter, applied on the sample grid |
| `src/daw/effects.py` | `delay`: a signal, later (echo and reverb are sums of delays) |
| `src/daw/keyboard.py` | `keyboard`: play a signal with your computer's keys, with a scope and a spectrogram |
| `src/daw/visualizer.py` | `visualize`: a window that plays a signal and shows it (needs `pip install 'daw[gui]'`) |
| `src/daw/analysis.py` | The numbers behind the spectrogram and the overview, with no GUI |
| `src/daw/live.py` | Real-time engine: notes rendered in pieces, mixed and sent to the speakers |
| `docs/` | How to use the library, with graphs; the code blocks are executed |
| `examples/` | Example scripts (`house.py`, `house_with_low_pass.py`, ...) |
| `tests/` | Pytest suite, golden files and guard tests |
| `tools/` | `mutation_check.py` (proves the tests catch bugs), `build_docs.py` (runs the docs) |
| `guidelines.md` | How we design, test, lint and commit, and the principles behind the library. Read before contributing |

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
| Format | `ruff format tests` |
| Type check | `mypy src/daw` |

Tool settings live in `pyproject.toml` (ruff, mypy, pytest).

`nix flake check` runs the tests, ruff, mypy and the docs check in the Nix
sandbox. GitHub Actions runs it on every push to `main` and every pull request
(`.github/workflows/ci.yml`); the slow mutation check runs nightly and on demand from
the Actions tab (`mutation.yml`).

Every change, new feature and fixed bug needs rigorous tests. The standard,
per-feature checklists and conventions are in [guidelines.md](guidelines.md).

## Running the examples

```sh
python examples/house.py
python examples/house_with_low_pass.py
python examples/ode_to_joy.py
python examples/echo_pulse.py       # asks how many seconds between the echoes
python examples/echo_melody.py      # three plucks: dry, with an echo, in a room
python examples/keyboard_synth.py   # a playable instrument (needs the gui extra)
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
