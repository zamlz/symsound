#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""A melody whose echoes bounce from ear to ear.

The three notes start in the left ear. Each echo is a delayed, quieter copy, and
the copies go to the right ear, then the left, then the right again. A
`StereoExpr` is just a left signal and a right signal, so each ear is built from
ordinary `delay` terms. Use headphones, or two speakers, to hear it.
"""

from symsound.constants import Note
from symsound.effects import delay
from symsound.envelope import ACTIVE_DURATION, adsr_envelope
from symsound.stereo import StereoExpr
from symsound.waves.basic import sine
from symsound.waves.utils import play

seconds = 4.0
pluck = adsr_envelope(0.005, 0.25, 0.0, 0.1).subs(ACTIVE_DURATION, 0.01)

# Six echoes, each 0.6 times as loud as the one before, 0.375 s apart
echo_spacing = 0.375
echo_feedback = 0.6
echo_repeats = 6


def note(pitch):
    """A pluck with a little second harmonic, turned down so the echoes never clip"""
    return 0.7 * (sine(pitch) + 0.3 * sine(2 * pitch)) * pluck


def ping_pong(signal, spacing, feedback, repeats):
    """The signal in the left ear, then `repeats` echoes `spacing` seconds
    apart, each `feedback` times as loud as the one before, alternating
    right, left, right..."""
    echoes = [feedback**k * delay(signal, k * spacing) for k in range(1, repeats + 1)]
    return StereoExpr(
        signal + sum(echoes[1::2]),  # echo 2, 4, 6: back in the left ear
        sum(echoes[::2]),  # echo 1, 3, 5: in the right ear
    )


# C, E, G, one every half second
melody = note(Note.C4) + delay(note(Note.E4), 0.5) + delay(note(Note.G4), 1.0)
bouncing = ping_pong(melody, echo_spacing, echo_feedback, echo_repeats)

# Only play when run as a script, so importing this file is silent
if __name__ == "__main__":
    play(bouncing, seconds)
