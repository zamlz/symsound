#!/usr/bin/env python3

import numpy as np

from daw.constants import Note
from daw.waves.basic import sin, sawtooth
from daw.waves.utils import compile, plot_signal
from daw.waves.time import TIME
from daw.sound import play, save

w1 = sin(0.4, amplitude=0.2)
w2 = sin(0.4, amplitude=0.2, time=TIME+1)
w3 = sin(0.4, amplitude=0.2, time=TIME+2)
w4 = sin(0.4, amplitude=0.2, time=TIME+3)

x = sin(Note.C5, amplitude=w1)
y = sin(Note.E5, amplitude=w2)
z = sin(Note.G5, amplitude=w3)
n = sin(Note.C6, amplitude=w4)

chord = x + y + z + n
audio = compile(chord, duration=5)

play(audio)
plot_signal(x)
