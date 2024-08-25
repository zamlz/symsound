#!/usr/bin/env python3

from daw.waves.basic import sine, square, sawtooth
from daw.waves.utils import compile, plot, WHITE_NOISE
from daw.sound import play, save
from daw.waves.time import BPM, TIME

# w = sine(440) + sine(480)

bpm = BPM(150)

kick_frequency = bpm.frequency(1)
kick_time_shift = TIME + bpm.time(0.5)
kick_decay = .5
kick_beat = 1 - (sawtooth(kick_frequency, time_shift=kick_time_shift) + 0.5)

kick = sine(40, amplitude=kick_beat**kick_decay)

hat_frequency = bpm.frequency(1)
hat_time_shift = TIME + bpm.time(1)
hat_decay = 20
hat_beat = 1 - (sawtooth(hat_frequency, time_shift=hat_time_shift) + 0.5)

hat = 0.25 * (hat_beat**hat_decay) * WHITE_NOISE


saw_frequency = bpm.frequency(4)
saw_time_shift = TIME + bpm.time(4)
saw_decay = 9
saw_beat = 1 - (sawtooth(saw_frequency, time_shift=saw_time_shift) + 0.5)

stutter_lfo = sine(bpm.frequency(4), amplitude=6) + 12.5
stutter = square(stutter_lfo) + 0.5
saw = 0.1*sawtooth(1000, amplitude=saw_beat**saw_decay) * stutter

w = kick + hat + saw

play(compile(w, 10))
save(compile(w, 30), "beat.wav")
