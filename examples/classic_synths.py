#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham

from symsound.envelope import ACTIVE_DURATION, adsr_envelope
from symsound.filters import low_pass_filter
from symsound.waves.basic import sawtooth, sine, square
from symsound.waves.time import BPM, TIME
from symsound.waves.utils import WHITE_NOISE, play, plot, save

w1 = sine(440)  # * adsr_envelope(.1, .3, .6, .2).subs(ACTIVE_DURATION, 1)
w2 = low_pass_filter(w1, 150 + 15*sine(1))

# play(w2, 4)
