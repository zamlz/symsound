#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham

import sympy as sp

from daw.filters import low_pass_filter
from daw.waves.basic import sawtooth, sine, square
from daw.waves.time import BPM, TIME
from daw.waves.utils import WHITE_NOISE, compile, play, plot, save

bpm = BPM(140)

kick_frequency = bpm.frequency(1)
kick_time_shift = TIME + bpm.time(0.5)
kick_beat = (1 - sawtooth(kick_frequency, time_shift=kick_time_shift)) / 2
kick_decay = .5

kick = (kick_beat**kick_decay) * sine(40)

hat_frequency = bpm.frequency(1)
hat_time_shift = TIME + bpm.time(1)
hat_beat = (1 - sawtooth(hat_frequency, time_shift=hat_time_shift)) / 2
hat_decay = 20

hat = 0.25 * (hat_beat**hat_decay) * WHITE_NOISE


saw_frequency = bpm.frequency(4)
saw_time_shift = TIME + bpm.time(4)
saw_beat = (1 - sawtooth(saw_frequency, time_shift=saw_time_shift)) / 2
saw_decay = 9

stutter_lfo = 6*sine(bpm.frequency(4)) + 12.5
stutter = square(stutter_lfo) + 1.0

saw = 0.1 * (saw_beat**saw_decay) * sawtooth(1000) * stutter

# The plain sum of the parts peaks at about 1.19 and would clip: scale the
# mix by 0.8 so the peak stays just under 1
master_gain = 0.8

dry = master_gain * (kick + hat + saw)

# The low-pass sweep lives in the middle third only. `sweep` rises smoothly
# from 0 to 1 and back to 0 across it, and is 0 everywhere else. The cutoff
# falls exponentially from 12 kHz to 150 Hz at the centre, then comes back up.
# `wet` fades the filtered sound in and out so the outer thirds are the dry
# sound exactly, with no filter state or phase shift left over.
duration = 30
third = duration / 3

sweep = sp.Piecewise(
    (
        sp.sin(sp.pi * (TIME - third) / third) ** 2,
        sp.And(TIME >= third, TIME <= 2 * third),
    ),
    (0.0, True),
)
cutoff = 12000 * sp.exp(-sp.log(12000 / 150) * sweep)
wet = sp.Min(1.0, 5 * sweep)

w = dry + wet * (low_pass_filter(dry, cutoff) - dry)

# Only play and save when run as a script, so the signals above can be
# imported (the docs do this to draw graphs of them)
if __name__ == "__main__":
    play(w, duration)
    save(w, duration, "beat.wav")
