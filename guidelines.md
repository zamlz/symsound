# Guidelines

How we build and test `daw`. Everything here is a working agreement: follow it
for every change, human or AI. The goal is simple: **a change that alters the
sound, the API or the numbers must fail a test, and a change that does not must
leave every test green.** We never want to find out by ear that something
regressed.

Contents

1. [Definition of done](#1-definition-of-done)
2. [Commands](#2-commands)
3. [Project layout](#3-project-layout)
4. [How we test](#4-how-we-test)
5. [Recipes by kind of feature](#5-recipes-by-kind-of-feature)
6. [Writing good tests here](#6-writing-good-tests-here)
7. [Bugs, limitations and strict xfails](#7-bugs-limitations-and-strict-xfails)
8. [Golden files](#8-golden-files)
9. [Guards that enforce all this](#9-guards-that-enforce-all-this)
10. [Mutation check](#10-mutation-check)
11. [Code style and dependencies](#11-code-style-and-dependencies)
12. [Commits](#12-commits)
13. [Known limitations register](#13-known-limitations-register)
14. [Writing the docs](#14-writing-the-docs)

---

## 1. Definition of done

A change is done only when **all** of these hold:

- [ ] `nix flake check` passes (pytest, ruff and mypy).
- [ ] Every new public function, class or constant has tests at the level
      described in [section 5](#5-recipes-by-kind-of-feature), not just a smoke
      test.
- [ ] Every bug fixed has a regression test that fails without the fix. Prove
      it: revert the fix, watch the test fail, restore the fix.
- [ ] Line **and** branch coverage of `src/daw` is still 100%.
- [ ] New examples are added to `tests/test_examples.py` and have a golden
      file ([section 8](#8-golden-files)).
- [ ] The README and this document are updated if behavior, commands or
      conventions changed.
- [ ] Every new public function or class is explained in `docs/` for someone
      who wants to *use* it, with an executable example and, for anything that
      makes or changes a signal, a graph (see [section 14](#14-writing-the-docs)).
      `tests/test_docs.py` fails on an undocumented public name, and the
      `docs` flake check fails if the docs are stale. If you change
      `examples/kick_with_low_pass.py`, update `docs/kick-with-low-pass-walkthrough.md`; the test requires
      it to quote the file.
- [ ] Nothing was validated only by "it ran". If it makes sound, the test
      measures the sound (peak level, pitch, silence where there should be
      silence, length), see [section 4](#4-how-we-test).

If something cannot be tested, say so in the pull request and record it in
the limitations register ([section 13](#13-known-limitations-register)).

## 2. Commands

Everything runs through the Nix dev shell, which provides Python, all
dependencies, ruff and `PYTHONPATH=src`.

| Task                                | Command                                                                                |
| ----------------------------------- | -------------------------------------------------------------------------------------- |
| Everything CI runs                  | `nix flake check`                                                                      |
| One check                           | `nix build .#checks.x86_64-linux.pytest` (or `ruff`, `mypy`)                           |
| Tests (fast loop)                   | `nix develop -c pytest`                                                                |
| Tests without the slow example runs | `nix develop -c pytest -m "not slow"`                                                  |
| One file or test                    | `nix develop -c pytest tests/test_envelope.py -k release`                              |
| Tests without coverage output       | `nix develop -c pytest --no-cov`                                                       |
| Lint                                | `nix develop -c ruff check .`                                                          |
| Format                              | `nix develop -c ruff format tests` (see [section 11](#11-code-style-and-dependencies)) |
| Types                               | `nix develop -c mypy src/daw`                                                          |
| Regenerate golden files             | `UPDATE_GOLDEN=1 nix develop -c pytest tests/test_examples.py`                         |
| Mutation check                      | `nix develop -c python tools/mutation_check.py`                                        |

Notes:

- Flakes only see files known to git. After creating a new file run
  `git add` (or `git add -N`) before `nix flake check`, or the check will not
  find it.
- `nix flake check` runs on a clean copy of the repository with
  `HYPOTHESIS_PROFILE=ci`, so property tests are deterministic there. Locally
  the default profile is randomized on purpose: it keeps finding new edge
  cases. If a randomized run fails, **keep the failing example** (Hypothesis
  prints it) and turn it into a permanent test.
- `ty` is deliberately not used; mypy is the type checker.

## 3. Project layout

```
src/daw/               the library
  constants.py         Note table, DEFAULT_SAMPLE_RATE
  envelope.py          adsr_envelope, ACTIVE_DURATION
  filters.py           low_pass_filter (first-order IIR on the sample grid)
  waves/basic.py       sine, square, triangle, sawtooth, sawtooth_v2, trigger
  waves/time.py        TIME symbol, BPM
  waves/utils.py       linspace, white_noise, compile, play, save, plot
examples/              runnable scripts; also our acceptance tests
docs/                  user documentation; every python block is tested
tests/
  conftest.py          fixtures, Hypothesis profiles
  helpers.py           evaluate() and independent reference implementations
  golden.py            golden-file support
  golden/*.npy         fingerprints of what the examples sound like
  test_*.py            one file per area (see below)
  test_policy.py       guards that enforce this document
  test_docs.py         runs the docs' code and checks they cover the API
tools/mutation_check.py  deliberately breaks the code to prove tests notice
```

Signals are SymPy expressions of one shared symbol, `daw.waves.time.TIME`.
They only become samples in `compile()`. Keep that separation: construction is
symbolic, evaluation is numeric.

## 4. How we test

A test that only calls a function and checks it does not crash proves almost
nothing here: a signal can run fine and still be wrong by a factor of 600
(this actually happened, see [section 7](#7-bugs-limitations-and-strict-xfails)).
Test at several layers, and use the layer that fits.

1. **Known values.** Hand-computed numbers at points that matter: the
   quarter, half and full period of a wave; the peak, midpoint and end of each
   envelope stage; one beat at 120 BPM is 0.5 s. These document intent and
   catch gross errors.
2. **Independent reference implementation.** Re-derive the expected output
   with plain NumPy/Python in `tests/helpers.py` (`ref_square`,
   `ref_triangle`, `ref_sawtooth`, `ref_adsr`) and compare over many inputs.
   The reference must be written **differently** from the code under test (for
   example the triangle reference uses `1 - 4x` / `4x - 3` pieces, the library
   uses `2|2x-1|-1`). A reference that copies the implementation can only
   copy its bugs.
3. **Mathematical properties** (Hypothesis). Statements that must hold for
   every input: output stays in range, waves are periodic, doubling the
   frequency doubles the rate, a time shift equals evaluating later, a round
   trip returns the input, linearity, monotonic segments, continuity.
4. **Closed forms.** When physics gives an exact answer, test against it: the
   RC low-pass step response is `1 - exp(-t/tau)`, the sine response has a known
   transient and gain `1/sqrt(1+(w*tau)^2)`, FM phase is `f0*t - depth*cos(...)/(2*pi*fm)`.
5. **Edge cases and degenerate parameters.** Zero attack, zero decay, zero
   release, zero-length notes, release during the attack, sustain 0 and 1,
   empty durations, negative values that must be rejected, exact stage
   boundaries.
6. **Measure the audio, do not eyeball it.** For anything that makes sound,
   assert on the samples: peak is at most 1.0, the signal is finite, notes
   have the expected pitch (count zero crossings or use an FFT), rests are
   exactly silent, the length matches the score, RMS is audible. This is what
   caught the ode_to_joy bug.
7. **Integration through the examples.** `tests/test_examples.py` runs every
   script end to end with audio stubbed out. They prove the whole pipeline
   (symbolic build, compile, play, save) still works together.
8. **Golden fingerprints.** Each audible example has a stored fingerprint of
   its signal. Any change to how an example sounds fails until a human has
   looked at it and regenerated the file on purpose ([section 8](#8-golden-files)).
9. **Regression tests.** Each fixed bug keeps a test named for what went
   wrong, with a comment explaining the bug.

Rules of thumb:

- Assert on behavior and numbers, not on how the expression is spelled.
  (One exception: guards against a known construction-time hazard, like the
  `TIME < 0` branch being dropped.)
- Prefer a few precise assertions over one vague one. A tolerance should be
  as tight as the arithmetic allows and have a reason.
- A test must be able to fail. When you add one, break the code on purpose and
  confirm. [Section 10](#10-mutation-check) automates this for core logic.
- Never use real audio devices, the network, or the wall clock.

## 5. Recipes by kind of feature

Use these as checklists. Each item is a test (or a parametrized group of
tests). Put them in the matching `tests/test_*.py` file.

### A new waveform (`waves/basic.py`)

- [ ] Matches an independent reference over random frequencies and times
      (away from discontinuities, see `away_from_edges`).
- [ ] Output stays within its documented range, including a dense sweep.
- [ ] Periodic with period `1/f`.
- [ ] `wave(2f)(t) == wave(f)(2t)`.
- [ ] `time_shift=TIME + d` equals evaluating at `t + d`.
- [ ] Default frequency is `Note.A4`; frequency may be an expression (FM): the
      phase is the integral of the frequency.
- [ ] Zero mean over whole periods (or document why not).
- [ ] Shape-specific facts: key points, slope, jump size, duty cycle.
- [ ] `free_symbols == {TIME}`.
- [ ] Add it to `WAVES` in `tests/test_basic_waves.py` so every generic
      property runs on it.

### An envelope or a time-varying modulator (`envelope.py`)

- [ ] Known value table through each stage.
- [ ] Matches an independent piecewise reference over random parameters.
- [ ] Bounds, silence before the start and after the end, continuity where
      the shape promises it.
- [ ] **Time shifting**: `subs(TIME, TIME - start)` leaves it silent before
      `start` and equal to the unshifted version afterwards. Songs are built
      this way.
- [ ] Every degenerate parameter (zero, extremes) and every invalid one.
- [ ] Release or end from every stage.
- [ ] Stacking many shifted copies (like `ode_to_joy`) stays bounded.

### A filter or other signal processor (`filters.py`)

- [ ] Step response and sine response against the closed form.
- [ ] Gain at DC, at the cutoff and far above it; phase lag; roll-off slope.
- [ ] Linearity and zero-in/zero-out.
- [ ] Works on every kind of signal, not just sines: square, triangle,
      noise, envelopes, sums, FM, constants. Compare against an independent
      simulation (`scipy.signal.lsim`, `scipy.integrate.solve_ivp`) and a
      plain-Python loop of the discretisation. Never test only the inputs
      that happen to work; coverage cannot tell you about the ones you skipped.
- [ ] Input types: ints, `sympy.Integer`, floats, expressions. Anything that
      fails is a strict xfail with the error type.
- [ ] Time-varying parameters compared to an ODE solver on the exact input.
- [ ] Time shifting (inside and outside the filter) and sample-rate
      independence; error shrinks at the expected order when the rate doubles.
- [ ] Invalid input (negative cutoff, a single time, an uneven grid) raises.

### A utility (`waves/utils.py`) or anything that touches files, audio or plots

- [ ] Shapes, lengths, dtypes and defaults, for random durations and rates.
- [ ] Round trips (`save` then `soundfile.read`) within the format's
      quantization error.
- [ ] Failure modes: bad paths, interrupted playback, clipping behavior.
- [ ] Audio goes through the `fake_sounddevice` fixture and plots through the
      Agg backend (`conftest.py`). Assert on what was sent to them.
- [ ] Optional dependencies stay optional (see the subprocess test that
      blocks `sounddevice`).

### A new constant or table (`constants.py`)

- [ ] Verify every entry against its definition (the `Note` table is checked
      against the equal-temperament formula for all 88 keys).
- [ ] Names, uniqueness, ordering and lookup round trips.

### A new example script (`examples/`)

- [ ] Add its name to `test_every_example_is_covered_by_this_file`.
- [ ] It runs without error, plays finite audible audio, and (if it saves a
      file) the file has the right format and length.
- [ ] Add a golden fingerprint ([section 8](#8-golden-files)).
- [ ] If it builds something nontrivial (a melody, a drum pattern), add
      behavior tests like `TestOdeToJoy`: pitches, silence, level, length.

### A change to `TIME`, `compile` or other core machinery

These affect everything. Run the full suite and the mutation check, and think
about which downstream assumptions (sympy simplification, lambdify, shifted
notes) the change could break.

## 6. Writing good tests here

- **One file per area**, named after what it tests (`test_envelope.py`).
  Group related tests in classes (`class TestSawtooth`).
- **Evaluate with `helpers.evaluate(expr, t)`**. It lambdifies like
  `compile()` and always returns an array, even for constants. Do not call
  `expr.subs(...).evalf()` in hot loops: it is very slow on large
  expressions (a 200-term `Sum` takes tens of seconds).
- **Hypothesis**: keep strategies realistic (`frequencies`, `stage`, `levels`
  in the existing files are good templates). Use `assume()` to skip
  mathematically ambiguous points such as the exact instant of a jump.
  Failures reproduce from Hypothesis' printed example.
- **Tolerances**: sympy and NumPy agree to about 1e-9 for these formulas.
  Steep envelopes (a 1 ms stage has slope 1000) amplify rounding, use
  `abs=1e-9` rather than the default `1e-12`. Sympy may rewrite
  `TIME - t0 <= x` as `TIME <= x + t0`, which rounds differently exactly on a
  boundary, so avoid sampling exactly on stage boundaries when comparing a
  shifted and an unshifted envelope.
- **Determinism**: call `np.random.seed(n)` before anything that draws noise.
  Never assert on unseeded random values.
- **Warnings are errors** (`filterwarnings = error` in `pyproject.toml`). If a
  third-party deprecation is unavoidable, ignore that exact message there,
  with a comment saying why.
- **Slow tests** that run whole examples carry `@pytest.mark.slow`
  (`pytestmark` in `test_examples.py`). Keep the rest fast.
- **No test may depend on another** or on execution order.

## 7. Bugs, limitations and strict xfails

When you find a bug and are not fixing it in the same change, record it as a
strict expected failure:

```python
@pytest.mark.xfail(strict=True, raises=TypeError, reason="why, and the workaround")
def test_the_correct_behavior():
    ...
```

`strict=True` is mandatory (enforced by `test_policy.py`). The test then
passes while the bug exists and **fails loudly the day someone fixes it**,
which tells them to delete the marker and keep the test. Pass `raises=` when
you know the exception type so a different failure is not hidden.

When you fix a bug:

1. Write a test that reproduces it and watch it fail.
2. Fix the code.
3. Watch the test pass, and keep it with a comment explaining the original
   bug. Example: `TIME` was declared `positive=True`, so sympy evaluated
   `TIME < 0` to `False` while building `adsr_envelope` and dropped the "silent
   before the note starts" branch. Shifted notes then ramped up backwards
   before they began, the notes summed to a peak of about 675 and
   `ode_to_joy` sounded like noise. The tests that now guard it are
   `TestTimeSymbol` in `test_time.py`, `TestTimeShifting` in
   `test_envelope.py` and `TestOdeToJoy` in `test_examples.py`.

## 8. Golden files

`tests/golden/<example>.npy` holds every 997th sample of the signal an example
plays. It is small, yet any audible change shifts it.

- A golden test failing means **the sound changed**. Decide whether that was
  intended.
- If it was intended, regenerate with
  `UPDATE_GOLDEN=1 nix develop -c pytest tests/test_examples.py`, listen to
  the example, and commit the new `.npy` in the same commit as the change that
  caused it, explaining why the sound changed.
- If it was not intended, you found a regression. Fix the code.
- Examples with random noise seed NumPy first (`run_example` does), so
  fingerprints are reproducible.

## 9. Guards that enforce all this

`tests/test_policy.py` fails when:

- a public function or class in `src/daw` is not referenced by any test;
- an example script has no tests or an audible example has no golden file;
- an `xfail` is not `strict=True`;
- a test is skipped unconditionally.

Coverage: `pyproject.toml` sets `fail_under = 100` for line and branch
coverage of `daw`. If a new branch is uncovered, add a test for it. Do not
lower the threshold, and do not add `# pragma: no cover` without a comment
explaining why that line cannot be tested.

These guards catch missing tests, not weak ones. Reviewing the quality of the
assertions against [section 4](#4-how-we-test) is still your job.

## 10. Mutation check

100% coverage proves code ran, not that tests would notice it being wrong.
`tools/mutation_check.py` injects about thirty deliberate bugs (an inverted
slope, a wrong sample rate, a dropped guard, a typo in the note table) one at
a time and confirms the suite fails for each.

```
nix develop -c python tools/mutation_check.py
```

- Every mutant must be **KILLED**. A **SURVIVED** line means a real bug could
  slip through: add the missing test, then re-run.
- Run it after changing core logic or adding a feature, and add mutants for
  new behavior in the `MUTANTS` list: `(label, file, old_text, new_text)`.
- The script restores every file it touches. If it is interrupted, check
  `git status` and `git diff src` before doing anything else.
- It takes several minutes, so it is not part of `nix flake check`.

## 11. Code style and dependencies

- **ruff** lints everything (`ruff check .`): pycodestyle, pyflakes, isort,
  bugbear, pyupgrade and simplify rules. flake8 is not used, ruff covers it.
  `SIM108` and `SIM300` are ignored on purpose: sympy comparisons read better
  as `TIME <= x`.
- **ruff format** is applied to `tests/`. `src/` has not been reformatted yet.
  Do not mix a reformat of `src/` into a functional change.
- **mypy** checks `src/daw`. New code in `src/daw` is typed.
- Match the surrounding code: naming, comments, and idiom.
- Example scripts may have unused imports (`examples/*.py` is exempt from
  `F401`/`E402`). The library may not.
- **Python 3.14** is the supported floor (`requires-python`). The flake uses
  the default `python3` of the pinned nixpkgs.
- **Dependencies live in two places.** Add a runtime dependency to
  `pyproject.toml` `dependencies` **and** to `pythonEnv` in `flake.nix`; a dev
  tool goes in the `develop` extras **and** `pythonEnv`. The flake checks fail
  otherwise.
- `sounddevice` is imported lazily inside `play()` on purpose: compiling,
  saving and testing must work with no audio backend.
- Update the pinned nixpkgs with `nix flake update`, then run
  `nix flake check` and the examples. Mixing a new host with an old pin once
  broke audio (glibc mismatch with `LD_LIBRARY_PATH`).

## 12. Commits

- Messages are short, lowercase and descriptive ("fixed shifted envelopes
  ramping up before they start"). Add a body when the reason is not obvious.
- One logical change per commit: tooling, a bug fix, a feature, a move. Keep
  a test with the change it protects, so every commit passes `nix flake check`.
- Commits are authored by the repository owner only, with **no
  `Co-Authored-By` trailers**, and are **not signed** (pass `--no-gpg-sign`;
  the owner's git config signs by default).
- Never commit `result` symlinks, `beat.wav` or other generated output (they
  are in `.gitignore`).
- Do not push without being asked.

## 13. Known limitations register

Each of these has a test. When you fix one, flip its test as described in
[section 7](#7-bugs-limitations-and-strict-xfails) and delete the entry.

| Limitation | Where | Test |
| --- | --- | --- |
| `compile()` of a constant expression returns a scalar instead of one value per sample, so `play`/`save` of pure silence or DC would fail | `waves/utils.py` | `test_constant_expression_is_broadcast_to_every_sample` (xfail) |
| A filtered expression can only be evaluated over a uniform sample grid, not at a single time; the filter starts at rest at the first sample | `filters.py` | `TestValidation` |
| `WHITE_NOISE` is one shared symbol, so every use in an expression sees the same samples | `waves/utils.py` | `test_white_noise_symbol_is_one_shared_signal` |
| `sawtooth_v2` is scaled by pi/2 and overshoots about 18% (Gibbs), unlike `sawtooth` which is within +/-1 | `waves/basic.py` | `TestSawtoothV2` |
| `examples/kick.py` and `kick_with_low_pass.py` peak at about 1.19, so it clips on playback and in saved files | `examples/kick.py` | golden file (`test_example_sound_has_not_changed`) |
| `adsr_envelope` uses one global `ACTIVE_DURATION` symbol, so each note needs its own `subs` | `envelope.py` | `TestReleaseFromAnyStage` |
| Phase integration is symbolic (`sp.integrate`), so frequency expressions SymPy cannot integrate fail, and large ones are slow | `waves/basic.py` | `test_frequency_modulation_integrates_the_frequency` |
| `adsr_envelope` raises a bare `Exception` for negative parameters | `envelope.py` | `TestValidation` |
| Flake checks only ran on x86_64-linux; other platforms are untested | `flake.nix` | n/a |

## 14. Writing the docs

`docs/` is for people who want to *use* the library. Explain what each piece
does and how to use it, not how it is implemented, and show it with graphs.

The pages are processed by [`md`](https://github.com/zamlz/mdutils) (an input
of our flake, available in the dev shell). A python block followed by an
`md-code` comment is run, and whatever it prints is written back under it:

~~~markdown
```python
from daw.waves.basic import sine
from daw.waves.utils import plot

plot(sine(3), duration=1.0, output_file="img/my-page-sine.png")
```
<!-- md-code: id="my-page-sine"; bin="python3" -->

![A 3 Hz sine](img/my-page-sine.png)
~~~

| Task | Command |
| --- | --- |
| Run every block and refresh output and graphs | `nix develop -c python tools/build_docs.py` |
| Check nothing is stale (what the flake runs) | `nix develop -c python tools/build_docs.py --check` |
| Fast structural checks only | `nix develop -c pytest tests/test_docs.py -m "not slow"` |

Rules, all enforced by `tests/test_docs.py` or the `docs` flake check:

- **Each block runs alone**, in a fresh `python3`, from the `docs/` directory.
  Import everything it uses. Paths are relative to `docs/` (`img/...`,
  `../examples/kick_with_low_pass.py`).
- **Graphs are made by the blocks**, with `plot(..., output_file=...)` and
  `spectrogram(..., output_file=...)`. Name images `img/<page>-<what>.png`, add
  alt text, and keep them 10 inches wide (the defaults). Commit the PNGs.
- **Use a graph for any signal**: a waveform to show shape, a spectrogram to
  show which frequencies there are and how they move (filters, FM, sweeps,
  notes). Use slow frequencies (a few Hz) when you want to see individual
  cycles, `start=` to look inside a long piece, and `stacked=True` with
  `labels` when signals differ in level.
- **Anything random must be seeded** (`np.random.seed(0)` before using
  `WHITE_NOISE`), or the image changes on every run and the check fails.
- **Output blocks are written by `md`**. Never edit the text under a block, and
  do not edit the `md-code-output` comments. Rerun the tool.
- A python block that should not be run (it needs speakers) is written
  ```` ```python no-run ````. The only other undirected blocks are quotes of
  `examples/kick_with_low_pass.py`, which must match the file exactly.
- A block that raises is recorded by `md` as a traceback; the check treats that
  as a failure.
- After changing the library, regenerate the docs and look at the diff of the
  images (open the PNGs): a graph that changed is a sound that changed.
