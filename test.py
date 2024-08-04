#!/usr/bin/env python3

import numpy as np

from daw.constants import Note
from daw.waves.basic import sin, sawtooth
from daw.waves.utils import compile
from daw.sound import play, save

x = sin(Note.C5, amplitude=0.2)
y = sin(Note.E5, amplitude=0.2)
z = sin(Note.G5, amplitude=0.2)
c_major = x + y + z
c_major_audio = compile(c_major, duration=1)

chromatic_wave = []
for note in Note:
    new_wave = compile(sin(note), duration=0.05)
    chromatic_wave.append(new_wave)

audio = np.concatenate(chromatic_wave)
audio += audio[::-1]
audio = np.concatenate([c_major_audio, audio])

play(audio)
save(audio, 'audio.wav')
