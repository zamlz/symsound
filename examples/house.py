#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham

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

# Each part is at most 1, so the plain sum clips (it peaks at about 1.19).
# Scaling the mix by 0.8 keeps the peak just under 1.
master_gain = 0.8

w = master_gain * (kick + hat + saw)

# Only play when run as a script, so importing this file is silent
if __name__ == "__main__":
    play(w, 10)
    save(w, 30, "beat.wav")
