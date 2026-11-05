#!/usr/bin/env python3

from daw.envelope import ACTIVE_DURATION, adsr_envelope
from daw.filters import low_pass_filter
from daw.waves.basic import sawtooth, sine, square
from daw.waves.time import BPM, TIME
from daw.waves.utils import WHITE_NOISE, play, plot, save

w1 = sine(440)  # * adsr_envelope(.1, .3, .6, .2).subs(ACTIVE_DURATION, 1)
w2 = low_pass_filter(w1, 150 + 15*sine(1))

# play(w2, 4)
