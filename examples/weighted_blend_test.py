#!/usr/bin/env python3

import numpy as np
import sympy as sp

from daw.constants import Note
from daw.envelope import adsr_envelope
from daw.waves.basic import sawtooth, sine, square, triangle
from daw.waves.time import TIME
from daw.waves.utils import play


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

play(w, 3)
