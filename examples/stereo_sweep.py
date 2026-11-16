#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""A held chord that sweeps from ear to ear.

The pan angle does not have to be a number: it can be an expression of `TIME`.
Here it is `pi/4 * sine(0.5)`, a slow sine that swings the sound from hard left
to hard right and back every two seconds. Use headphones, or two speakers, to
hear it.
"""

import sympy as sp

from symsound.constants import Note
from symsound.effects import pan
from symsound.envelope import ACTIVE_DURATION, adsr_envelope
from symsound.waves.basic import sine
from symsound.waves.utils import play

seconds = 4.0
sweep_rate = 0.5  # sweeps per second: one trip across and back every two seconds
chord_level = 0.3  # three notes add up, so each is turned down


def chord_with_fade(length):
    """C, E and G held together, faded in and out so there is no click"""
    held = adsr_envelope(0.1, 0.0, 1.0, 0.3).subs(ACTIVE_DURATION, length - 0.3)
    return chord_level * (sine(Note.C4) + sine(Note.E4) + sine(Note.G4)) * held


sweeping = pan(chord_with_fade(seconds), sp.pi / 4 * sine(sweep_rate))

# Only play when run as a script, so importing this file is silent
if __name__ == "__main__":
    play(sweeping, seconds)
