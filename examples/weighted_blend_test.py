#!/usr/bin/env python3

import numpy as np
import sympy as sp

from symsound.constants import Note
from symsound.envelope import adsr_envelope
from symsound.waves.basic import sawtooth, sine, square, triangle
from symsound.waves.time import TIME
from symsound.waves.utils import play


def weighted_blend(w1, w2, blend_duration, sharpness=1):
    return sp.Piecewise(
        (
            w1 * (1 - TIME / blend_duration) ** sharpness
            + w2 * (TIME / blend_duration) ** sharpness,
            TIME <= blend_duration
        ),
        (w2, TIME > blend_duration)
    )


w1 = square(frequency=440)
w2 = 0.5*sine(frequency=440)
w = weighted_blend(w1, w2, 1, 1)

# Only play when run as a script, so importing this file is silent
if __name__ == "__main__":
    play(w, 3)
