#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""Three plucked notes placed across the ears: C hard left, E in the centre and
G hard right.

`pan(signal, angle)` puts a mono signal between the ears. The angle is in
radians from the centre: 0 is the middle, pi/4 is all right and -pi/4 all left.
`delay` starts E half a second after C, and G a second after it. Use
headphones, or two speakers, to hear it.
"""

import sympy as sp

from symsound.constants import Note
from symsound.effects import delay, pan
from symsound.envelope import ACTIVE_DURATION, adsr_envelope
from symsound.stereo import StereoExpr
from symsound.waves.basic import sine
from symsound.waves.utils import play

seconds = 2.0
pluck = adsr_envelope(0.005, 0.25, 0.0, 0.1).subs(ACTIVE_DURATION, 0.01)


def note(pitch):
    """A pluck with a little second harmonic, turned down so three never clip"""
    return 0.7 * (sine(pitch) + 0.3 * sine(2 * pitch)) * pluck


def mix(*pairs):
    """Stereo signals played together: the lefts add up, and so do the rights"""
    return StereoExpr(
        sum(pair.left for pair in pairs), sum(pair.right for pair in pairs)
    )


# C left, E centre, G right, one every half second
placed = mix(
    pan(note(Note.C4), -sp.pi / 4),
    pan(delay(note(Note.E4), 0.5), 0),
    pan(delay(note(Note.G4), 1.0), sp.pi / 4),
)

# Only play when run as a script, so importing this file is silent
if __name__ == "__main__":
    play(placed, seconds)
