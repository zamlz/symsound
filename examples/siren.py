#!/usr/bin/env python3

import numpy as np
import sympy as sp

from daw.constants import Note
from daw.waves.basic import sine, sawtooth, square, triangle
from daw.waves.utils import compile, plot_signal
from daw.waves.time import TIME
from daw.sound import play, save


def a5():
    note_rate = [
        (12, Note.A5),
        (5, Note.C5),
        (2.5, Note.F5),
    ]

    #freq = square(.5, 220) + 440
    wave = 0
    for rate, note in note_rate:
        freq = square(rate, note) + note
        wave += sine(freq, 1/(len(note_rate)+1))

    wave += sine(Note.A2, 0.2)
    return wave

def g5():
    note_rate = [
        (12, Note.G5),
        (5, Note.B5),
        (2.5, Note.E5),
    ]

    #freq = square(.5, 220) + 440
    wave = 0
    for rate, note in note_rate:
        freq = square(rate, note) + note
        wave += sine(freq, 1/(len(note_rate)+1))

    wave += sine(Note.D2, 0.2)
    return wave

def c5():
    note_rate = [
        (12, Note.C5),
        (5, Note.E5),
        (2.5, Note.G5),
    ]

    #freq = square(.5, 220) + 440
    wave = 0
    for rate, note in note_rate:
        freq = square(rate, note) + note
        wave += sine(freq, 1/(len(note_rate)+1))

    wave += sine(Note.C2, 0.2)
    return wave

audio_list = [
    compile(a5(), 4),
    compile(g5(), 2),
    compile(c5(), 4),
    compile(a5(), 4),
    compile(g5(), 2),
    compile(c5(), 4),
    compile(a5(), 1),
    compile(g5(), 3),
    compile(c5(), 5),
    compile(a5(), 5),
    compile(g5(), 1),
    compile(c5(), 2),
    compile(a5(), 1),
]
play(np.concatenate(audio_list))
