#!/usr/bin/env python3

from daw.waves.basic import sine, square, sawtooth
from daw.waves.utils import compile, plot_signal, WHITE_NOISE
from daw.sound import play, save
from daw.waves.time import TIME

# w = sine(440) + sine(480)

kick_f = 2.5
kick_decay = 1
kick_beat = (1 - (sawtooth(kick_f, time_shift=TIME+(1/(kick_f*2))) + 0.5))**kick_decay
kick = sine(30, amplitude=kick_beat)

hat_f = 2.5
hat_decay = 20
hat_beat = (1 - (sawtooth(hat_f, time_shift=TIME+(2/(hat_f*2))) + 0.5))**hat_decay
hat = 0.5 * hat_beat * WHITE_NOISE

w = kick + hat
play(compile(w, 10))
