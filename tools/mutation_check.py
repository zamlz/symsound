#!/usr/bin/env python3
"""Prove the tests notice bugs by deliberately introducing them

Each mutant replaces one piece of source text with a plausible mistake and
runs the test suite. A mutant is KILLED when the suite fails, SURVIVED when
every test still passes, which means a real bug of that shape could slip
through. See guidelines.md for when to run this and how to react.

Usage: python tools/mutation_check.py [label-substring ...]
"""

import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (label, file, text to replace, replacement). Replacements apply once.
MUTANTS = [
    # envelope
    (
        "envelope attack slope",
        "src/daw/envelope.py",
        "attack_func = TIME / attack",
        "attack_func = TIME / (attack * 2)",
    ),
    (
        "envelope decay slope sign",
        "src/daw/envelope.py",
        "decay_to_sustain_slope = (sustain - 1.0) / decay",
        "decay_to_sustain_slope = (1.0 - sustain) / decay",
    ),
    (
        "envelope release slope sign",
        "src/daw/envelope.py",
        "release_slope = -1 * env_at_release / release",
        "release_slope = env_at_release / release",
    ),
    (
        "envelope drops negative-time guard",
        "src/daw/envelope.py",
        "(0.0, TIME < 0),\n",
        "",
    ),
    (
        "envelope zero-attack level",
        "src/daw/envelope.py",
        "attack_func = 1.0",
        "attack_func = 0.5",
    ),
    (
        "envelope ignores negative release",
        "src/daw/envelope.py",
        "attack < 0 or decay < 0 or sustain < 0 or release < 0",
        "attack < 0 or decay < 0 or sustain < 0",
    ),
    # time
    (
        "TIME declared positive again",
        "src/daw/waves/time.py",
        "sp.symbols('time', real=True)",
        "sp.symbols('time', real=True, positive=True)",
    ),
    (
        "BPM.time inverted",
        "src/daw/waves/time.py",
        "return beats / self._bps",
        "return beats * self._bps",
    ),
    (
        "BPM.beat inverted",
        "src/daw/waves/time.py",
        "return self._bps * time",
        "return time / self._bps",
    ),
    (
        "BPM seconds per minute",
        "src/daw/waves/time.py",
        "self._bpm / 60",
        "self._bpm / 61",
    ),
    # waves
    (
        "square becomes sine",
        "src/daw/waves/basic.py",
        "return sp.sign(sp.sin(2 * sp.pi * phase))",
        "return sp.sin(2 * sp.pi * phase)",
    ),
    (
        "triangle offset",
        "src/daw/waves/basic.py",
        "return 2 * sp.Abs(2 * (phase % 1) - 1) - 1",
        "return 2 * sp.Abs(2 * (phase % 1) - 1)",
    ),
    (
        "sawtooth floor",
        "src/daw/waves/basic.py",
        "sp.floor(phase + 0.5)",
        "sp.floor(phase)",
    ),
    (
        "sawtooth_v2 sign",
        "src/daw/waves/basic.py",
        "(-1)**(i+1)",
        "(-1)**i",
    ),
    (
        "sawtooth_v2 default resolution",
        "src/daw/waves/basic.py",
        "resolution: int = 200",
        "resolution: int = 100",
    ),
    (
        "sine drops the factor of 2",
        "src/daw/waves/basic.py",
        "return sp.sin(2 * sp.pi * phase)\n\n\ndef square",
        "return sp.sin(sp.pi * phase)\n\n\ndef square",
    ),
    (
        "trigger boundary",
        "src/daw/waves/basic.py",
        "(1.0, TIME >= trigger_time)",
        "(1.0, TIME > trigger_time)",
    ),
    # utils
    (
        "linspace includes endpoint",
        "src/daw/waves/utils.py",
        "endpoint=False",
        "endpoint=True",
    ),
    (
        "white noise range",
        "src/daw/waves/utils.py",
        "np.random.uniform(-1.0, 1.0, num_samples)",
        "np.random.uniform(0.0, 1.0, num_samples)",
    ),
    (
        "compile argument order",
        "src/daw/waves/utils.py",
        "sp.lambdify((TIME, WHITE_NOISE)",
        "sp.lambdify((WHITE_NOISE, TIME)",
    ),
    (
        "play never waits",
        "src/daw/waves/utils.py",
        "        sd.wait()\n",
        "        pass\n",
    ),
    (
        "play wrong sample rate",
        "src/daw/waves/utils.py",
        "sd.play(compiled_wave, samplerate=sample_rate)",
        "sd.play(compiled_wave, samplerate=sample_rate * 2)",
    ),
    (
        "save wrong sample rate",
        "src/daw/waves/utils.py",
        "sf.write(output_file, compiled_wave, sample_rate)",
        "sf.write(output_file, compiled_wave, sample_rate * 2)",
    ),
    (
        "plot axis label",
        "src/daw/waves/utils.py",
        "last.set_xlabel('Time [s]' if bpm is None else 'Beat [b]')",
        "last.set_xlabel('Beat [b]')",
    ),
    (
        "plot ignores bpm duration",
        "src/daw/waves/utils.py",
        "return amount if bpm is None else bpm.time(amount)",
        "return amount",
    ),
    # plotting
    (
        "plot window ignores start length",
        "src/daw/waves/utils.py",
        "total = start + duration",
        "total = duration",
    ),
    (
        "plot window starts at zero",
        "src/daw/waves/utils.py",
        "first = int(round(start * sample_rate))",
        "first = 0",
    ),
    (
        "plot start not converted from beats",
        "src/daw/waves/utils.py",
        "wave, seconds(start), seconds(duration), sample_rate",
        "wave, start, seconds(duration), sample_rate",
    ),
    (
        "plot panels do not share time",
        "src/daw/waves/utils.py",
        "len(waves), 1, sharex=True,",
        "len(waves), 1, sharex=False,",
    ),
    (
        "plot accepts the wrong number of labels",
        "src/daw/waves/utils.py",
        "if labels is not None and len(labels) != len(waves):",
        "if False:",
    ),
    (
        "plot output does not make directories",
        "src/daw/waves/utils.py",
        "path.parent.mkdir(parents=True, exist_ok=True)",
        "pass",
    ),
    (
        "spectrogram colour range",
        "src/daw/waves/utils.py",
        "vmin=loudest - SPECTROGRAM_RANGE_DB,",
        "vmin=loudest - 60,",
    ),
    (
        "spectrogram frequency limit ignored",
        "src/daw/waves/utils.py",
        "max_frequency if max_frequency is not None else sample_rate / 2",
        "sample_rate / 2",
    ),
    (
        "spectrogram accepts short windows",
        "src/daw/waves/utils.py",
        "if len(samples) < SPECTROGRAM_WINDOW:",
        "if len(samples) < 0:",
    ),
    (
        "spectrogram frequency axis",
        "src/daw/waves/utils.py",
        "extent=(start, start + len(samples) / sample_rate, 0, sample_rate / 2),",
        "extent=(start, start + len(samples) / sample_rate, 0, sample_rate),",
    ),
    # filters and constants
    (
        "low-pass time constant (constant cutoff)",
        "src/daw/filters.py",
        "k = np.pi * float(fc) / sample_rate",
        "k = 2 * np.pi * float(fc) / sample_rate",
    ),
    (
        "low-pass feedback coefficient",
        "src/daw/filters.py",
        "b, a = k / (1 + k), (1 - k) / (1 + k)",
        "b, a = k / (1 + k), 1 - k",
    ),
    (
        "low-pass does not start at rest",
        "src/daw/filters.py",
        "zi=[b * x[0]]",
        "zi=[b * x[0] + 0.5]",
    ),
    (
        "low-pass time constant (varying cutoff)",
        "src/daw/filters.py",
        "w = 2 * np.pi * fc",
        "w = np.pi * fc",
    ),
    (
        "low-pass step size (varying cutoff)",
        "src/daw/filters.py",
        "half_step = 0.5 / sample_rate",
        "half_step = 1.0 / sample_rate",
    ),
    (
        "low-pass implicit term",
        "src/daw/filters.py",
        "(1 + half_step * cur_w)",
        "(1 + half_step * prev_w)",
    ),
    (
        "low-pass error sign",
        "src/daw/filters.py",
        "prev_w * (prev_x - prev_y)",
        "prev_w * (prev_x + prev_y)",
    ),
    (
        "low-pass allows negative cutoff arrays",
        "src/daw/filters.py",
        "if (fc < 0).any():",
        "if (fc < -2).any():",
    ),
    (
        "low-pass accepts zero sample rate",
        "src/daw/filters.py",
        "if sample_rate <= 0:",
        "if sample_rate < 0:",
    ),
    (
        "low-pass accepts uneven grids",
        "src/daw/filters.py",
        "if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-6, atol=0):",
        "if steps[0] <= 0:",
    ),
    (
        "low-pass sample rate from grid",
        "src/daw/filters.py",
        "return apply_low_pass(x, fc, 1 / steps[0])",
        "return apply_low_pass(x, fc, 2 / steps[0])",
    ),
    (
        "low-pass allows negative cutoff expressions",
        "src/daw/filters.py",
        "if cutoff.is_negative:",
        "if False:",
    ),
    (
        "concert pitch",
        "src/daw/constants.py",
        "A4 = 440.00",
        "A4 = 441.00",
    ),
    (
        "typo in the note table",
        "src/daw/constants.py",
        "D4 = 293.66",
        "D4 = 293.00",
    ),
    (
        "default sample rate",
        "src/daw/constants.py",
        "DEFAULT_SAMPLE_RATE = 44100",
        "DEFAULT_SAMPLE_RATE = 48000",
    ),
]


DOCS_FRESHNESS_TEST = (
    "tests/test_docs.py::test_docs_are_up_to_date_with_what_their_code_produces"
)


def run_suite():
    env = dict(os.environ, HYPOTHESIS_PROFILE="ci")
    command = [sys.executable, "-m", "pytest", "-x", "-q", "--no-cov"]
    command += ["-p", "no:cacheprovider", "tests"]
    # The docs freshness test rebuilds every graph, so it would "kill" any
    # mutant that changes a picture. That says nothing about whether the unit
    # tests notice, which is what this tool measures.
    command += ["--deselect", DOCS_FRESHNESS_TEST]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    failures = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(("FAILED", "ERROR"))
    ]
    return result.returncode != 0, failures[0] if failures else ""


def main(filters):
    # Let the `finally` below restore the source if this is killed
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(130))
    mutants = [m for m in MUTANTS if not filters or any(f in m[0] for f in filters)]
    problems = []
    for label, relative, old, new in mutants:
        path = ROOT / relative
        original = path.read_text()
        if old not in original:
            print(f"STALE     {label}: text not found in {relative}", flush=True)
            problems.append(label)
            continue
        path.write_text(original.replace(old, new, 1))
        try:
            killed, first_failure = run_suite()
        finally:
            path.write_text(original)
        status = "KILLED  " if killed else "SURVIVED"
        print(f"{status}  {label}   {first_failure[:80]}", flush=True)
        if not killed:
            problems.append(label)
    print(f"\n{len(mutants) - len(problems)}/{len(mutants)} mutants killed")
    if problems:
        print("needs attention:", ", ".join(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
