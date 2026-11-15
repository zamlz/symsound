#!/usr/bin/env python3
"""Stereo: three plucked notes placed across the ears, an echo that bounces
from ear to ear, and a chord that sweeps from side to side.

Each version plays in turn. Use headphones, or two speakers, to hear it.

* placed: `pan` puts C hard left (-pi/4), E in the centre (0) and G hard
  right (pi/4).
* ping-pong: the melody starts in the left ear and each echo goes to the
  opposite ear from the one before.
* sweep: a held chord whose pan angle is a slow sine, so it travels across
  the ears and back (`pan` takes an expression of `TIME` as the angle).
"""

import numpy as np
import sympy as sp

from daw.constants import Note
from daw.effects import delay, pan
from daw.envelope import ACTIVE_DURATION, adsr_envelope
from daw.stereo import StereoExpr
from daw.waves.basic import sine
from daw.waves.utils import compile, play

pluck = adsr_envelope(0.005, 0.25, 0.0, 0.1).subs(ACTIVE_DURATION, 0.01)

# Ping-pong: six echoes, each 0.6 times as loud, 0.375 s apart
echo_spacing = 0.375
echo_feedback = 0.6
echo_repeats = 6

# Sweep: the angle goes from ear to ear and back once every two seconds
sweep_rate = 0.5
sweep_seconds = 4.0
chord_level = 0.3  # three notes add up, so each is turned down


def note(pitch):
    """A pluck with a little second harmonic"""
    return (sine(pitch) + 0.3 * sine(2 * pitch)) * pluck


def mix(*pairs):
    """Stereo signals played together: the lefts add up, and so do the rights"""
    return StereoExpr(
        sum(pair.left for pair in pairs), sum(pair.right for pair in pairs)
    )


def ping_pong(signal, spacing, feedback, repeats):
    """The signal in the left ear, then `repeats` echoes `spacing` seconds
    apart, each `feedback` times as loud as the one before, alternating
    right, left, right..."""
    echoes = [feedback**k * delay(signal, k * spacing) for k in range(1, repeats + 1)]
    return StereoExpr(
        signal + sum(echoes[1::2]),  # echo 2, 4, 6: back in the left ear
        sum(echoes[::2]),  # echo 1, 3, 5: in the right ear
    )


def chord_with_fade(seconds):
    """C, E and G held together, faded in and out so there is no click"""
    held = adsr_envelope(0.1, 0.0, 1.0, 0.3).subs(ACTIVE_DURATION, seconds - 0.3)
    return chord_level * (sine(Note.C4) + sine(Note.E4) + sine(Note.G4)) * held


# C left, E centre, G right, one every half second
placed = mix(
    pan(note(Note.C4), -sp.pi / 4),
    pan(delay(note(Note.E4), 0.5), 0),
    pan(delay(note(Note.G4), 1.0), sp.pi / 4),
)

melody = note(Note.C4) + delay(note(Note.E4), 0.5) + delay(note(Note.G4), 1.0)

# Name: (signal, seconds to play)
versions = {
    "placed": (placed, 2.0),
    "ping-pong": (ping_pong(melody, echo_spacing, echo_feedback, echo_repeats), 4.0),
    "sweep": (
        pan(chord_with_fade(sweep_seconds), sp.pi / 4 * sine(sweep_rate)),
        sweep_seconds,
    ),
}

# Only play when run as a script, so importing this file is silent
if __name__ == "__main__":
    for name, (signal, seconds) in versions.items():
        # Turn the whole thing down if it would clip
        peak = float(np.abs(compile(signal, seconds)).max())
        print(f"{name}: playing for {seconds:.0f} s")
        play(
            StereoExpr(
                signal.left / max(peak, 1.0), signal.right / max(peak, 1.0)
            ),
            seconds,
        )
