#!/usr/bin/env python3

import numpy as np
import sympy as sp

from daw.constants import Note
from daw.envelope import adsr_envelope
from daw.waves.basic import sine, sawtooth, square, triangle
from daw.waves.utils import compile, plot_signal
from daw.waves.time import TIME
from daw.sound import play, save

ode_to_joy = [
    (Note.FS4, 1),
    (Note.FS4, 1),
    (Note.G4, 1),
    (Note.A4, 1),
    (Note.A4, 1),
    (Note.G4, 1),
    (Note.FS4, 1),
    (Note.E4, 1),
    (Note.D4, 1),
    (Note.D4, 1),
    (Note.E4, 1),
    (Note.FS4, 1),
    (Note.FS4, 1.5),
    (Note.E4, .5),
    (Note.E4, 1),
]

def get_duration(notes):
    return sum(d for _, d in notes)

def wave(note):
    a = sine(4 * (note / Note.FS4), amplitude=(0.5 * (note / Note.FS4)))
    x = sine(note, amplitude=.5) * adsr_envelope(.2, .2, 0.5, .2, .5)
    return x

BPM = 60
bpm = lambda x: x * (60 / BPM)

audio = [compile(wave(note), bpm(duration)) for note, duration in ode_to_joy]
audio = np.concatenate(audio)
play(audio)
