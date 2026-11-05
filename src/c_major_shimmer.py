#!/usr/bin/env python3

import numpy as np

from daw.constants import Note
from daw.waves.basic import sawtooth, sine, square, triangle
from daw.waves.time import TIME
from daw.waves.utils import compile, play, plot, save

w1 = 0.2 * triangle(1)
w2 = 0.2 * triangle(1, time_shift=TIME+0.15)
w3 = 0.2 * triangle(1, time_shift=TIME+0.30)
w4 = 0.2 * triangle(1, time_shift=TIME+0.45)

x = w1 * sine(Note.C5)
y = w2 * sine(Note.E5)
z = w3 * sine(Note.G5)
n = w3 * sine(Note.C6)

chord = x + y + z + n
play(chord, duration=5)
