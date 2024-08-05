#!/usr/bin/env python3

import numpy as np

from daw.constants import Note
from daw.waves.basic import sine, sawtooth, square, triangle
from daw.waves.utils import compile, plot_signal
from daw.waves.time import TIME
from daw.sound import play, save

freq = square(1, 220) + 440
wave = sine(freq)
play(compile(wave, 10))
