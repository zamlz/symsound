#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""Prove the tests notice bugs by deliberately introducing them

Each mutant replaces one piece of source text with a plausible mistake and
runs the test suite. A mutant is KILLED when the suite fails, SURVIVED when
every test still passes, which means a real bug of that shape could slip
through. See guidelines.md for when to run this and how to react.

Each mutant first runs only the tests that cover its source file, which takes
seconds. If those miss it, the whole suite runs before the mutant is declared a
survivor, so the shortcut can never hide a real gap, it can only be slower.

Usage: python tools/mutation_check.py [--rotate N] [label-substring ...]

  --rotate N   run only N mutants, a different N each day (for nightly runs
               that cannot afford all of them); every mutant comes round
"""

import datetime
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (label, file, text to replace, replacement). Replacements apply once.
MUTANTS = [
    # effects
    (
        "delay is not silenced before it arrives",
        "src/symsound/effects/timing.py",
        "return sp.Piecewise((_shift(signal, amount), TIME >= amount), (0, True))",
        "return _shift(signal, amount)",
    ),
    (
        "delay arrives at the wrong end of the mask",
        "src/symsound/effects/timing.py",
        "(_shift(signal, amount), TIME >= amount)",
        "(_shift(signal, amount), TIME > amount)",
    ),
    (
        "delay shifts the wrong way",
        "src/symsound/effects/timing.py",
        "return expression.subs(TIME, TIME - seconds)",
        "return expression.subs(TIME, TIME + seconds)",
    ),
    (
        "delay lets a filter start before its sound",
        "src/symsound/effects/timing.py",
        "started * _shift(signal, seconds)",
        "_shift(signal, seconds)",
    ),
    (
        "delay leaves a sweeping cutoff unshifted",
        "src/symsound/effects/timing.py",
        "_shift(cutoff, seconds), TIME",
        "cutoff, TIME",
    ),
    (
        "delay only shifts the top level of a filtered mix",
        "src/symsound/effects/timing.py",
        "return expression.func(*[_shift(arg, seconds) for arg in expression.args])",
        "return expression.subs(TIME, TIME - seconds)",
    ),
    (
        "delay accepts a negative amount",
        "src/symsound/effects/timing.py",
        "if amount.is_negative:",
        "if amount.is_negative is None:",
    ),
    (
        "delay accepts an imaginary amount",
        "src/symsound/effects/timing.py",
        "if not amount.is_number or not amount.is_real:",
        "if not amount.is_number:",
    ),
    (
        "delay of nothing is no longer the signal itself",
        "src/symsound/effects/timing.py",
        "if amount.is_zero:",
        "if amount.is_zero and False:",
    ),
    # envelope
    (
        "envelope attack slope",
        "src/symsound/envelope.py",
        "attack_func = TIME / attack",
        "attack_func = TIME / (attack * 2)",
    ),
    (
        "envelope decay slope sign",
        "src/symsound/envelope.py",
        "decay_to_sustain_slope = (sustain - 1.0) / decay",
        "decay_to_sustain_slope = (1.0 - sustain) / decay",
    ),
    (
        "envelope release slope sign",
        "src/symsound/envelope.py",
        "release_slope = -1 * env_at_release / release",
        "release_slope = env_at_release / release",
    ),
    (
        "envelope drops negative-time guard",
        "src/symsound/envelope.py",
        "(0.0, TIME < 0),\n",
        "",
    ),
    (
        "envelope zero-attack level",
        "src/symsound/envelope.py",
        "attack_func = 1.0",
        "attack_func = 0.5",
    ),
    (
        "envelope ignores negative release",
        "src/symsound/envelope.py",
        "attack < 0 or decay < 0 or sustain < 0 or release < 0",
        "attack < 0 or decay < 0 or sustain < 0",
    ),
    # time
    (
        "TIME declared positive again",
        "src/symsound/waves/time.py",
        "sp.symbols('time', real=True)",
        "sp.symbols('time', real=True, positive=True)",
    ),
    (
        "BPM.time inverted",
        "src/symsound/waves/time.py",
        "return beats / self._bps",
        "return beats * self._bps",
    ),
    (
        "BPM.beat inverted",
        "src/symsound/waves/time.py",
        "return self._bps * time",
        "return time / self._bps",
    ),
    (
        "BPM seconds per minute",
        "src/symsound/waves/time.py",
        "self._bpm / 60",
        "self._bpm / 61",
    ),
    # waves
    (
        "square becomes sine",
        "src/symsound/waves/basic.py",
        "return sp.sign(sp.sin(2 * sp.pi * phase))",
        "return sp.sin(2 * sp.pi * phase)",
    ),
    (
        "triangle offset",
        "src/symsound/waves/basic.py",
        "return 2 * sp.Abs(2 * (phase % 1) - 1) - 1",
        "return 2 * sp.Abs(2 * (phase % 1) - 1)",
    ),
    (
        "sawtooth floor",
        "src/symsound/waves/basic.py",
        "sp.floor(phase + 0.5)",
        "sp.floor(phase)",
    ),
    (
        "sawtooth_v2 sign",
        "src/symsound/waves/basic.py",
        "(-1)**(i+1)",
        "(-1)**i",
    ),
    (
        "sawtooth_v2 default resolution",
        "src/symsound/waves/basic.py",
        "resolution: int = 200",
        "resolution: int = 100",
    ),
    (
        "sine drops the factor of 2",
        "src/symsound/waves/basic.py",
        "return sp.sin(2 * sp.pi * phase)\n\n\ndef square",
        "return sp.sin(sp.pi * phase)\n\n\ndef square",
    ),
    (
        "trigger boundary",
        "src/symsound/waves/basic.py",
        "(1.0, TIME >= trigger_time)",
        "(1.0, TIME > trigger_time)",
    ),
    # utils
    (
        "linspace includes endpoint",
        "src/symsound/waves/utils.py",
        "endpoint=False",
        "endpoint=True",
    ),
    (
        "white noise range",
        "src/symsound/waves/utils.py",
        "np.random.uniform(-1.0, 1.0, num_samples)",
        "np.random.uniform(0.0, 1.0, num_samples)",
    ),
    (
        "compile argument order",
        "src/symsound/waves/utils.py",
        "sp.lambdify((TIME, WHITE_NOISE)",
        "sp.lambdify((WHITE_NOISE, TIME)",
    ),
    (
        "compile does not repeat constants",
        "src/symsound/waves/utils.py",
        "np.array(np.broadcast_to(np.asarray(samples, dtype=float), times.shape))",
        "np.asarray(samples, dtype=float)",
    ),
    (
        "play never waits",
        "src/symsound/waves/utils.py",
        "        sd.wait()\n",
        "        pass\n",
    ),
    (
        "play wrong sample rate",
        "src/symsound/waves/utils.py",
        "sd.play(compiled_wave, samplerate=sample_rate)",
        "sd.play(compiled_wave, samplerate=sample_rate * 2)",
    ),
    (
        "save wrong sample rate",
        "src/symsound/waves/utils.py",
        "sf.write(output_file, compiled_wave, sample_rate)",
        "sf.write(output_file, compiled_wave, sample_rate * 2)",
    ),
    (
        "plot axis label",
        "src/symsound/waves/utils.py",
        "last.set_xlabel('Time [s]' if bpm is None else 'Beat [b]')",
        "last.set_xlabel('Beat [b]')",
    ),
    (
        "plot ignores bpm duration",
        "src/symsound/waves/utils.py",
        "return amount if bpm is None else bpm.time(amount)",
        "return amount",
    ),
    # plotting
    (
        "plot window ignores start length",
        "src/symsound/waves/utils.py",
        "total = start + duration",
        "total = duration",
    ),
    (
        "plot window starts at zero",
        "src/symsound/waves/utils.py",
        "first = int(round(start * sample_rate))",
        "first = 0",
    ),
    (
        "plot start not converted from beats",
        "src/symsound/waves/utils.py",
        "wave, seconds(start), seconds(duration), sample_rate",
        "wave, start, seconds(duration), sample_rate",
    ),
    (
        "plot panels do not share time",
        "src/symsound/waves/utils.py",
        "len(lines), 1, sharex=True,",
        "len(lines), 1, sharex=False,",
    ),
    (
        "plot accepts the wrong number of labels",
        "src/symsound/waves/utils.py",
        "if labels is not None and len(labels) != len(waves):",
        "if False:",
    ),
    (
        "plot output does not make directories",
        "src/symsound/waves/utils.py",
        "path.parent.mkdir(parents=True, exist_ok=True)",
        "pass",
    ),
    (
        "spectrogram colour range",
        "src/symsound/waves/utils.py",
        "vmin=loudest - SPECTROGRAM_RANGE_DB,",
        "vmin=loudest - 60,",
    ),
    (
        "spectrogram frequency limit ignored",
        "src/symsound/waves/utils.py",
        "max_frequency if max_frequency is not None else sample_rate / 2",
        "sample_rate / 2",
    ),
    (
        "spectrogram accepts short windows",
        "src/symsound/waves/utils.py",
        "if len(samples) < SPECTROGRAM_WINDOW:",
        "if len(samples) < 0:",
    ),
    (
        "spectrogram frequency axis",
        "src/symsound/waves/utils.py",
        "extent=(start, start + len(samples) / sample_rate, 0, sample_rate / 2),",
        "extent=(start, start + len(samples) / sample_rate, 0, sample_rate),",
    ),
    # filters and constants
    (
        "low-pass time constant (constant cutoff)",
        "src/symsound/filters.py",
        "k = np.pi * float(fc) / sample_rate",
        "k = 2 * np.pi * float(fc) / sample_rate",
    ),
    (
        "low-pass feedback coefficient",
        "src/symsound/filters.py",
        "b, a = k / (1 + k), (1 - k) / (1 + k)",
        "b, a = k / (1 + k), 1 - k",
    ),
    (
        "low-pass does not start at rest",
        "src/symsound/filters.py",
        "zi=[b * x[0]]",
        "zi=[b * x[0] + 0.5]",
    ),
    (
        "low-pass time constant (varying cutoff)",
        "src/symsound/filters.py",
        "w = 2 * np.pi * fc",
        "w = np.pi * fc",
    ),
    (
        "low-pass step size (varying cutoff)",
        "src/symsound/filters.py",
        "half_step = 0.5 / sample_rate",
        "half_step = 1.0 / sample_rate",
    ),
    (
        "low-pass implicit term",
        "src/symsound/filters.py",
        "(1 + half_step * cur_w)",
        "(1 + half_step * prev_w)",
    ),
    (
        "low-pass error sign",
        "src/symsound/filters.py",
        "prev_w * (prev_x - prev_y)",
        "prev_w * (prev_x + prev_y)",
    ),
    (
        "low-pass allows negative cutoff arrays",
        "src/symsound/filters.py",
        "if (fc < 0).any():",
        "if (fc < -2).any():",
    ),
    (
        "low-pass accepts zero sample rate",
        "src/symsound/filters.py",
        "if sample_rate <= 0:",
        "if sample_rate < 0:",
    ),
    (
        "low-pass accepts uneven grids",
        "src/symsound/filters.py",
        "if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-6, atol=0):",
        "if steps[0] <= 0:",
    ),
    (
        "low-pass sample rate from grid",
        "src/symsound/filters.py",
        "return apply_low_pass(x, fc, 1 / steps[0])",
        "return apply_low_pass(x, fc, 2 / steps[0])",
    ),
    (
        "low-pass allows negative cutoff expressions",
        "src/symsound/filters.py",
        "if cutoff.is_negative:",
        "if False:",
    ),
    (
        "concert pitch",
        "src/symsound/constants.py",
        "A4 = 440.00",
        "A4 = 441.00",
    ),
    (
        "typo in the note table",
        "src/symsound/constants.py",
        "D4 = 293.66",
        "D4 = 293.00",
    ),
    (
        "default sample rate",
        "src/symsound/constants.py",
        "DEFAULT_SAMPLE_RATE = 44100",
        "DEFAULT_SAMPLE_RATE = 48000",
    ),
    # live
    (
        "live note time does not start at zero",
        "src/symsound/live.py",
        "times = (first + np.arange(count)) / self.sample_rate",
        "times = np.arange(count) / self.sample_rate",
    ),
    (
        "live release measured in samples, not seconds",
        "src/symsound/live.py",
        "self.position, count, self.released_at / rate",
        "self.position, count, self.released_at",
    ),
    (
        "live empty chunk ends a release",
        "src/symsound/live.py",
        "silent = count > 0 and not samples.any()",
        "silent = not samples.any()",
    ),
    (
        "live fade does not fall",
        "src/symsound/live.py",
        "np.clip(1 - gone / fade, 0.0, 1.0)",
        "np.clip(1 + gone / fade, 0.0, 1.0)",
    ),
    (
        "live key auto-repeat restarts the note",
        "src/symsound/live.py",
        (
            "if any(v.key == key and v.released_at is None for v in self._voices):"
            "\n                return"
        ),
        "if False:\n                return",
    ),
    (
        "live mixer ignores gain",
        "src/symsound/live.py",
        "return self.gain * mix",
        "return mix",
    ),
    (
        "live finished notes are kept",
        "src/symsound/live.py",
        "self._voices = [v for v in self._voices if not v.finished]",
        "self._voices = list(self._voices)",
    ),
    (
        "live ring buffer forgets the order",
        "src/symsound/live.py",
        "indexes = np.arange(end - count, end) % self.capacity",
        "indexes = np.arange(count) % self.capacity",
    ),
    (
        "live output does not feed the tap",
        "src/symsound/live.py",
        "        if self.tap is not None:\n            self.tap.write(samples)",
        "        if self.tap is None:\n            self.tap = None",
    ),
    # analysis
    (
        "spectrogram scale is not decibels",
        "src/symsound/analysis.py",
        "np.clip(1 + decibels / range_db, 0.0, 1.0)",
        "np.clip(1 + decibels / (2 * range_db), 0.0, 1.0)",
    ),
    (
        "spectrogram rows skip the loudest bin",
        "src/symsound/analysis.py",
        "mixed[:, row] = np.maximum(mixed[:, row], covered)",
        "mixed[:, row] = mixed[:, row]",
    ),
    (
        "spectrogram columns lag the signal",
        "src/symsound/analysis.py",
        "padded = np.concatenate([np.zeros(window - hop), x])",
        "padded = np.concatenate([np.zeros(window), x])",
    ),
    (
        "spectrogram row frequencies are linear",
        "src/symsound/analysis.py",
        "return float(low * (high / low) ** (row / (rows - 1)))",
        "return float(low + (high - low) * row / (rows - 1))",
    ),
    (
        "peak envelope loses the lows",
        "src/symsound/analysis.py",
        "low = np.array([x[a:b].min() for a, b in zip(starts, stops, strict=True)])",
        "low = np.array([x[a:b].max() for a, b in zip(starts, stops, strict=True)])",
    ),
    # player
    (
        "player does not stop at the end",
        "src/symsound/live.py",
        (
            "            if self._position >= len(self.samples):\n"
            "                self._playing = False"
        ),
        (
            "            if self._position > len(self.samples):\n"
            "                self._playing = False"
        ),
    ),
    (
        "player play from the end does not restart",
        "src/symsound/live.py",
        (
            "            if self._position >= len(self.samples):\n"
            "                self._position = 0"
        ),
        (
            "            if self._position > len(self.samples):\n"
            "                self._position = 0"
        ),
    ),
    (
        "player seek is not clamped",
        "src/symsound/live.py",
        "self._position = min(max(target, 0), len(self.samples))",
        "self._position = target",
    ),
    (
        "player moves while paused",
        "src/symsound/live.py",
        "            if not self._playing:\n                return out",
        (
            "            if not self._playing:\n"
            "                self._position += count\n"
            "                return out"
        ),
    ),
    # visualizer
    (
        "seek bar ignores where you click",
        "src/symsound/visualizer.py",
        "return min(max((x - left) / width, 0.0), 1.0)",
        "return min(max((x - left) / width / 2, 0.0), 1.0)",
    ),
    (
        "dragging does not pause",
        "src/symsound/visualizer.py",
        "                self._resume = player.playing\n                player.pause()",
        "                self._resume = player.playing",
    ),
    (
        "unmapped key quits",
        "src/symsound/visualizer.py",
        "        if key in (pygame.K_ESCAPE, pygame.K_q):",
        "        if key not in (pygame.K_SPACE, pygame.K_r, pygame.K_HOME):",
    ),
    (
        "shift does not take the big step",
        "src/symsound/visualizer.py",
        "step = BIG_STEP if modifiers & pygame.KMOD_SHIFT else SMALL_STEP",
        "step = SMALL_STEP",
    ),
    (
        "spectrogram is drawn upside down",
        "src/symsound/_draw.py",
        "make_surface(picture[:, ::-1, :])",
        "make_surface(picture)",
    ),
    (
        "time format drops the tenths",
        "src/symsound/visualizer.py",
        'return f"{minutes}:{rest // 10:02d}.{rest % 10}"',
        'return f"{minutes}:{rest // 10:02d}.0"',
    ),
    # live spectrogram
    (
        "live spectrogram window does not end on a hop",
        "src/symsound/analysis.py",
        "samples = stream.latest(span + spare)[:span]",
        "samples = stream.latest(span)",
    ),
    (
        "live spectrogram ignores a stalled display",
        "src/symsound/analysis.py",
        "fresh = min(fresh, room)",
        "fresh = fresh",
    ),
    (
        "live spectrogram keeps every column",
        "src/symsound/analysis.py",
        "np.concatenate([self._levels, columns])[-self.max_columns :]",
        "np.concatenate([self._levels, columns])",
    ),
    # keyboard
    (
        "keyboard note off by an octave",
        "src/symsound/keyboard.py",
        'index = NOTES.index(Note[f"C{octave}"]) + semitone',
        'index = NOTES.index(Note[f"C{octave}"]) + semitone + 12',
    ),
    (
        "keyboard black keys wrong",
        "src/symsound/keyboard.py",
        "BLACK = {1, 3, 6, 8, 10, 13, 15}",
        "BLACK = {1, 3, 6, 8, 10, 13}",
    ),
    (
        "keyboard z goes up",
        "src/symsound/keyboard.py",
        "self.octave = max(self.octave - 1, FIRST_OCTAVE)",
        "self.octave = min(self.octave + 1, LAST_OCTAVE)",
    ),
    (
        "keyboard auto-repeat restarts the note",
        "src/symsound/keyboard.py",
        "if note is None or finger in self.sounding:",
        "if note is None:",
    ),
    (
        "keyboard notes stick when focus is lost",
        "src/symsound/keyboard.py",
        "            self.lift_all()  # the key-up would go to another window",
        "            pass",
    ),
    (
        "keyboard mouse does not slide between notes",
        "src/symsound/keyboard.py",
        'if semitone != self.sounding["mouse"][1]:',
        "if False:",
    ),
    (
        "keyboard white key under the black one",
        "src/symsound/keyboard.py",
        "for key in reversed(_piano_keys(rect)):",
        "for key in _piano_keys(rect):",
    ),
    (
        "keyboard display does not make the sound when silent",
        "src/symsound/keyboard.py",
        "            self.ring.write(self.mixer.render(round(seconds * rate)))",
        "            pass",
    ),
    (
        "keyboard accepts a signal with no envelope",
        "src/symsound/keyboard.py",
        "        if not instrument.source(note).has_release:",
        "        if False:",
    ),
    (
        "keyboard checks the envelope even when told not to",
        "src/symsound/keyboard.py",
        "    if require_envelope:\n        _check_envelope(instrument)\n",
        "    if True:\n        _check_envelope(instrument)\n",
    ),
    # drawing
    (
        "spectrogram colours reversed",
        "src/symsound/_draw.py",
        "return _PALETTE[np.round(clipped * 255).astype(int)]",
        "return _PALETTE[255 - np.round(clipped * 255).astype(int)]",
    ),
    (
        "scope is upside down",
        "src/symsound/_draw.py",
        "points = [(x + i, middle - h) for i, h in enumerate(heights)]",
        "points = [(x + i, middle + h) for i, h in enumerate(heights)]",
    ),
]


# The tests that cover each source file. A mutant is tried against these first
FAST_TESTS = {
    "src/symsound/envelope.py": ["tests/test_envelope.py"],
    "src/symsound/waves/time.py": ["tests/test_time.py"],
    "src/symsound/waves/basic.py": ["tests/test_basic_waves.py"],
    "src/symsound/waves/utils.py": ["tests/test_utils.py", "tests/test_plotting.py"],
    "src/symsound/filters.py": ["tests/test_filters.py"],
    "src/symsound/effects/timing.py": ["tests/test_effects.py"],
    "src/symsound/effects/spatial.py": ["tests/test_spatial.py"],
    "src/symsound/constants.py": ["tests/test_constants.py"],
    "src/symsound/live.py": ["tests/test_live.py"],
    "src/symsound/analysis.py": ["tests/test_analysis.py"],
    "src/symsound/visualizer.py": ["tests/test_visualizer.py"],
    "src/symsound/keyboard.py": ["tests/test_keyboard.py"],
    "src/symsound/_draw.py": ["tests/test_visualizer.py", "tests/test_keyboard.py"],
}

# Fails for every mutant by design (the mutated text is no longer in the file)
STALE_TEXT_TEST = (
    "tests/test_mutation_tool.py::test_every_mutant_text_is_still_in_the_source"
)

DOCS_FRESHNESS_TEST = (
    "tests/test_docs.py::test_docs_are_up_to_date_with_what_their_code_produces"
)


def run_suite(tests=None):
    """Run `tests` (default: the whole suite). Returns (failed, first failure)"""
    env = dict(os.environ, HYPOTHESIS_PROFILE="ci")
    command = [sys.executable, "-m", "pytest", "-x", "-q", "--no-cov"]
    command += ["-p", "no:cacheprovider", *(tests or ["tests"])]
    # The docs freshness test rebuilds every graph, so it would "kill" any
    # mutant that changes a picture, and the stale-text test fails for any
    # mutant at all. Neither says whether the unit tests notice a bug, which
    # is what this tool measures.
    command += ["--deselect", DOCS_FRESHNESS_TEST, "--deselect", STALE_TEXT_TEST]
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


def rotation(mutants, count, today=None):
    """`count` mutants for today; consecutive days cover the list in turn"""
    if count >= len(mutants):
        return mutants
    day = (today or datetime.date.today()).toordinal()
    return [mutants[(day * count + i) % len(mutants)] for i in range(count)]


def main(argv):
    # Let the `finally` below restore the source if this is killed
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(130))
    rotate = None
    if "--rotate" in argv:
        at = argv.index("--rotate")
        rotate = int(argv[at + 1])
        argv = argv[:at] + argv[at + 2 :]
    mutants = [m for m in MUTANTS if not argv or any(f in m[0] for f in argv)]
    if rotate:
        mutants = rotation(mutants, rotate)
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
            killed, first_failure = run_suite(FAST_TESTS.get(relative))
            note = ""
            if not killed and relative in FAST_TESTS:
                # The usual tests missed it: ask the whole suite before giving up
                killed, first_failure = run_suite()
                note = " (only the full suite noticed)"
        finally:
            path.write_text(original)
        status = "KILLED  " if killed else "SURVIVED"
        print(f"{status}  {label}   {first_failure[:80]}{note}", flush=True)
        if not killed:
            problems.append(label)
    print(f"\n{len(mutants) - len(problems)}/{len(mutants)} mutants killed")
    if problems:
        print("needs attention:", ", ".join(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
