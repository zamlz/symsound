#!/usr/bin/env python3

import numpy as np

from daw.constants import Note
from daw.waves.basic import sine, sawtooth, square, triangle
from daw.waves.utils import compile, plot
from daw.waves.time import TIME
from daw.sound import play, save

w1 = triangle(1, amplitude=0.2)
w2 = triangle(1, amplitude=0.2, time_shift=TIME+0.15)
w3 = triangle(1, amplitude=0.2, time_shift=TIME+0.30)
w4 = triangle(1, amplitude=0.2, time_shift=TIME+0.45)

x = sine(Note.C5, amplitude=w1)
y = sine(Note.E5, amplitude=w2)
z = sine(Note.G5, amplitude=w3)
n = sine(Note.C6, amplitude=w4)

chord = x + y + z + n
audio = compile(chord, duration=5)

play(audio)
#plot(chord)
