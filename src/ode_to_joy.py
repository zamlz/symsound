#!/usr/bin/env python3

from collections import defaultdict

import numpy as np
import sympy as sp

from daw.constants import Note
from daw.envelope import ACTIVE_DURATION, adsr_envelope
from daw.waves.basic import sine, sawtooth, square, triangle, trigger
from daw.waves.utils import play, save
from daw.waves.time import BPM, TIME

ode_to_joy_notes = [
    (Note.FS4, 0.5),
    (None, 0.5),
    (Note.FS4, 0.5),
    (None, 0.5),
    (Note.G4, 0.5),
    (None, 0.5),
    (Note.A4, 0.5),
    (None, 0.5),
    (Note.A4, 0.5),
    (None, 0.5),
    (Note.G4, 0.5),
    (None, 0.5),
    (Note.FS4, 0.5),
    (None, 0.5),
    (Note.E4, 0.5),
    (None, 0.5),
    (Note.D4, 0.5),
    (None, 0.5),
    (Note.D4, 0.5),
    (None, 0.5),
    (Note.E4, 0.5),
    (None, 0.5),
    (Note.FS4, 0.5),
    (None, 0.5),
    (Note.FS4, 0.5),
    (None, 1.0),
    (Note.E4, 0.5),
    (Note.E4, 0.5),
    (None, 1.5),
    (Note.FS4, 0.5),
    (None, 0.5),
    (Note.FS4, 0.5),
    (None, 0.5),
    (Note.G4, 0.5),
    (None, 0.5),
    (Note.A4, 0.5),
    (None, 0.5),
    (Note.A4, 0.5),
    (None, 0.5),
    (Note.G4, 0.5),
    (None, 0.5),
    (Note.FS4, 0.5),
    (None, 0.5),
    (Note.E4, 0.5),
    (None, 0.5),
    (Note.D4, 0.5),
    (None, 0.5),
    (Note.D4, 0.5),
    (None, 0.5),
    (Note.E4, 0.5),
    (None, 0.5),
    (Note.FS4, 0.5),
    (None, 0.5),
    (Note.E4, 0.5),
    (None, 1.0),
    (Note.D4, 0.5),
    (Note.D4, 0.5),
    (None, 0.5),
]

def get_duration(notes):
    return sum(d for _, d in notes)

def wave(note):
    vibrato = 20*sine(bpm.frequency(10))*sine(bpm.frequency(1))
    x = sine(note + vibrato) * adsr_envelope(.1, .1, 0.5, .1)
    return x

bpm = BPM(160)

beat = 0
ode_to_joy = 0
poly = False

for note, duration in ode_to_joy_notes:
    if note is not None:
        if poly:
            ode_to_joy *= (1 - trigger(bpm.time(beat)))
        ode_to_joy += wave(note).subs(
            ACTIVE_DURATION,
            bpm.time(duration)
        ).subs(
            TIME,
            TIME - bpm.time(beat)
        )
    beat += duration

play(ode_to_joy, bpm.time(beat))
