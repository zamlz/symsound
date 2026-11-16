#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham

import numpy as np

from symsound.constants import Note
from symsound.waves.basic import sawtooth, sine, square, triangle
from symsound.waves.time import TIME
from symsound.waves.utils import compile, play, plot, save

w1 = 0.2 * triangle(1)
w2 = 0.2 * triangle(1, time_shift=TIME+0.15)
w3 = 0.2 * triangle(1, time_shift=TIME+0.30)
w4 = 0.2 * triangle(1, time_shift=TIME+0.45)

x = w1 * sine(Note.C5)
y = w2 * sine(Note.E5)
z = w3 * sine(Note.G5)
n = w3 * sine(Note.C6)

chord = x + y + z + n
# Only play when run as a script, so importing this file is silent
if __name__ == "__main__":
    play(chord, duration=5)
