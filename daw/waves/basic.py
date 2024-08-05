from typing import Union

import sympy as sp
from sympy.core.expr import Expr

from daw.constants import Note
from daw.waves.time import TIME


def sine(
        frequency: Union[Expr, float, int] = Note.A4,
        amplitude: Union[Expr, float, int] = 0.5,
        time: Union[Expr, float, int] = TIME
) -> Expr:
    phase = 2 * sp.pi * frequency * time
    return amplitude * sp.sin(phase)


def square(
        frequency: Union[Expr, float, int] = Note.A4,
        amplitude: Union[Expr, float, int] = 0.5,
        time: Union[Expr, float, int] = TIME
) -> Expr:
    phase = 2 * sp.pi * frequency * time
    return amplitude * sp.sign(sp.sin(phase))


def triangle(
        frequency: Union[Expr, float, int] = Note.A4,
        amplitude: Union[Expr, float, int] = 0.5,
        time: Union[Expr, float, int] = TIME
) -> Expr:
    phase = frequency * time
    return amplitude * 2 * sp.Abs(2 * (phase % 1) - 1) - amplitude


def sawtooth(
        frequency: Union[Expr, float, int] = Note.A4,
        amplitude: Union[Expr, float, int] = 0.5,
        time: Union[Expr, float, int] = TIME
) -> Expr:
    phase = frequency * time
    return amplitude * 2 * (phase - sp.floor(phase + 0.5))
