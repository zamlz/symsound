#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""Three plucked notes, heard dry, with an echo, and in a room.

`delay` places the notes (E half a second after C, G a second after it), and
the same `delay` builds the echo and the reverb out of copies of the melody.
Each version plays in turn, so you can hear what the copies add.
"""

import numpy as np

from symsound.constants import Note
from symsound.effects import delay
from symsound.envelope import ACTIVE_DURATION, adsr_envelope
from symsound.waves.basic import sine
from symsound.waves.utils import compile, play

pluck = adsr_envelope(0.005, 0.25, 0.0, 0.1).subs(ACTIVE_DURATION, 0.01)

# Echo: six copies, each half as loud, a little under half a second apart
echo_spacing = 0.375
echo_feedback = 0.5
echo_repeats = 6

# Reverb: eight echo spacings, in seconds, with no common factor so the
# repeats never line up into one ringing pitch
reverb_spacings = (0.0253, 0.0269, 0.0289, 0.0307, 0.0322, 0.0338, 0.0353, 0.0367)
reverb_decay = 2.0  # seconds for the tail to fall 60 dB
reverb_level = 1.5


def note(pitch):
    """A pluck with a little second harmonic"""
    return (sine(pitch) + 0.3 * sine(2 * pitch)) * pluck


def echo(signal, spacing, feedback, repeats):
    """The signal, then `repeats` copies of it `spacing` seconds apart, each
    `feedback` times as loud as the one before"""
    return signal + sum(
        feedback**k * delay(signal, k * spacing) for k in range(1, repeats + 1)
    )


def reverb(signal, decay, level):
    """The signal plus a tail that falls 60 dB in `decay` seconds: echoes at
    each of several uneven spacings, which blur into one smooth tail"""
    wet = 0
    for spacing in reverb_spacings:
        # Each trip around one of these loops loses this much, so the copies
        # are a thousandth as loud after `decay` seconds
        gain = 10 ** (-3 * spacing / decay)
        wet += sum(
            gain**k * delay(signal, k * spacing)
            for k in range(1, int(decay / spacing) + 1)
        )
    return signal + level * wet / np.sqrt(len(reverb_spacings))


# C, E, G, one every half second
melody = note(Note.C4) + delay(note(Note.E4), 0.5) + delay(note(Note.G4), 1.0)

# Name: (signal, seconds to play)
versions = {
    "dry": (melody, 2.0),
    "echo": (echo(melody, echo_spacing, echo_feedback, echo_repeats), 4.0),
    "reverb": (reverb(melody, reverb_decay, reverb_level), 4.0),
}

# Only play when run as a script, so importing this file is silent
if __name__ == "__main__":
    for name, (signal, seconds) in versions.items():
        # The copies add up, so turn the whole thing down if it would clip
        peak = float(np.abs(compile(signal, seconds)).max())
        print(f"{name}: playing for {seconds:.0f} s")
        play(signal / max(peak, 1.0), seconds)
