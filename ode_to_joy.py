#!/usr/bin/env python3

import numpy as np
import sympy as sp

from daw.constants import Note
from daw.waves.basic import sine, sawtooth, square, triangle
from daw.waves.utils import compile, plot_signal
from daw.waves.time import TIME
from daw.sound import play, save


def auto_piecewise(notes):
    t = 0
    piecewise_args = []
    for note, duration in notes:
        piecewise_args.append((note, sp.And(TIME >= t, TIME < t + duration)))
        t += duration
    return sp.Piecewise(*piecewise_args)


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
    x = sine(note, amplitude=a)
    return x

otj = auto_piecewise(ode_to_joy)
audio = compile(wave(otj), duration=get_duration(ode_to_joy))

play(audio)
