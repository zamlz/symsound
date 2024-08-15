from typing import Union

import sympy as sp
from sympy.core.expr import Expr

from daw.constants import Note
from daw.waves.time import TIME


def sine(
        frequency: Union[Expr, float] = Note.A4,
        amplitude: Union[Expr, float] = 0.5,
        time_shift: Union[Expr, float] = TIME,
) -> Expr:
    phase = sp.integrate(frequency, TIME).subs(TIME, time_shift)
    return amplitude * sp.sin(2 * sp.pi * phase)


def square(
        frequency: Union[Expr, float, int] = Note.A4,
        amplitude: Union[Expr, float, int] = 0.5,
        time_shift: Union[Expr, float] = TIME,
) -> Expr:
    phase = sp.integrate(frequency, TIME).subs(TIME, time_shift)
    return amplitude * sp.sign(sp.sin(2 * sp.pi * phase))


def triangle(
        frequency: Union[Expr, float, int] = Note.A4,
        amplitude: Union[Expr, float, int] = 0.5,
        time_shift: Union[Expr, float] = TIME,
) -> Expr:
    phase = sp.integrate(frequency, TIME).subs(TIME, time_shift)
    return amplitude * 2 * sp.Abs(2 * (phase % 1) - 1) - amplitude


def sawtooth(
        frequency: Union[Expr, float, int] = Note.A4,
        amplitude: Union[Expr, float, int] = 0.5,
        time_shift: Union[Expr, float] = TIME,
) -> Expr:
    phase = sp.integrate(frequency, TIME).subs(TIME, time_shift)
    return amplitude * 2 * (phase - sp.floor(phase + 0.5))
