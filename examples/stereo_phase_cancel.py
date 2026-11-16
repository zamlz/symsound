#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""Phase cancellation: two 440 Hz sines, added in a mono signal and heard
separately in stereo.

Four versions of 5 seconds each, one after the other:

1. Two sines in phase, added together and played in mono: they line up, so the
   sum is a 440 Hz sine.
2. Two sines out of phase, added together and played in mono: the second sine is
   shifted by half a period (1/880 s), a phase shift of 180 degrees, so it is
   the first one upside down. They cancel exactly and this version is SILENT.
3. Two sines in phase, each in its own stereo channel: you hear a sine.
4. Two sines out of phase, each in its own stereo channel: you still HEAR a sine
   in each ear. The channels are not added by the signal. They go to your left
   and right ears (or speakers) separately, so nothing cancels. They would
   cancel only where something adds them: a mono mix of this signal, or two
   speakers whose sound meets in the air at a point equally far from both (and a
   room is never perfect).

Each sine has an amplitude of 0.5, so that two of them added together reach 1
and do not clip.

Run from the repository root:  nix develop -c python <this file>
"""

import numpy as np

from symsound.stereo import StereoExpr
from symsound.waves.basic import sine
from symsound.waves.time import TIME
from symsound.waves.utils import compile, play

FREQUENCY = 440
SECONDS = 5
SAMPLE_RATE = 44100

HALF_A_PERIOD = 1 / (2 * FREQUENCY)  # seconds: half a cycle is 180 degrees

first = 0.5 * sine(FREQUENCY)
second_in_phase = 0.5 * sine(FREQUENCY)
# sine(frequency, time_shift) is the sine at a shifted time: a delay of half a
# period turns the wave upside down
second_out_of_phase = 0.5 * sine(FREQUENCY, TIME - HALF_A_PERIOD)

versions = {
    "1. two sines in phase, added, mono": first + second_in_phase,
    "2. two sines out of phase, added, mono": first + second_out_of_phase,
    "3. two sines in phase, one per stereo channel": StereoExpr(
        first, second_in_phase
    ),
    "4. two sines out of phase, one per stereo channel": StereoExpr(
        first, second_out_of_phase
    ),
}


def peak(samples):
    return float(np.abs(samples).max())


if __name__ == "__main__":
    for name, signal in versions.items():
        samples = compile(signal, SECONDS, SAMPLE_RATE)
        print(f"{name}:")
        if samples.ndim == 1:
            print(f"  peak {peak(samples):.3g}")
        else:
            left, right = samples[:, 0], samples[:, 1]
            print(f"  left peak {peak(left):.3g}, right peak {peak(right):.3g}")
            print(f"  (added together they would peak at {peak(left + right):.3g})")
        play(signal, SECONDS, SAMPLE_RATE)
