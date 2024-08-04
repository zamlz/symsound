import abc

import sympy as sp

from daw.constants import time




def sine_wave(frequency: int, amplitude: float = 0.5):
    return amplitude * sp.sin(2 * sp.pi * frequency * time)


def sawtooth_wave(frequency: int, amplitude: float = 0.5):
    return 2*amplitude * (time*frequency - sp.floor(time*frequency + 0.5))
