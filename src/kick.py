#!/usr/bin/env python3

from daw.waves.basic import sine, square, sawtooth
from daw.waves.utils import compile, plot_signal, WHITE_NOISE
from daw.sound import play, save
from daw.waves.time import TIME

# w = sine(440) + sine(480)

kick_f = 2.5
kick_decay = .5
kick_beat = (1 - (sawtooth(kick_f, time_shift=TIME+(1/(kick_f*2))) + 0.5))**kick_decay
kick = sine(40, amplitude=kick_beat) + 0.01*sawtooth(42, amplitude=kick_beat**4)

hat_f = 2.5
hat_decay = 20
hat_beat = (1 - (sawtooth(hat_f, time_shift=TIME+(2/(hat_f*2))) + 0.5))**hat_decay
hat = 0.25 * hat_beat * WHITE_NOISE


saw_f = 2.5 * 0.25
saw_decay = 9
saw_beat = (1 - (sawtooth(saw_f, time_shift=TIME+(2/(saw_f*2))) + 0.5))**saw_decay
stutter_lfo = sine(2.5 * 0.25, amplitude=6) + 12.5
stutter = square(stutter_lfo) + 0.5
saw = 0.1*sawtooth(1000, amplitude=saw_beat) * stutter

w = kick + hat + saw
plot_signal(w, 30)
# play(compile(w, 30))
save(compile(w, 30), "beat.wav")
