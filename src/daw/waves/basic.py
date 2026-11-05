from typing import Union

import sympy as sp
from sympy.core.expr import Expr

from daw.constants import Note
from daw.waves.time import TIME


def sine(
        frequency: Expr | float = Note.A4,
        time_shift: Expr | float = TIME,
) -> Expr:
    phase = sp.integrate(frequency, TIME).subs(TIME, time_shift)
    return sp.sin(2 * sp.pi * phase)


def square(
        frequency: Expr | float | int = Note.A4,
        time_shift: Expr | float = TIME,
) -> Expr:
    phase = sp.integrate(frequency, TIME).subs(TIME, time_shift)
    return sp.sign(sp.sin(2 * sp.pi * phase))


def triangle(
        frequency: Expr | float | int = Note.A4,
        time_shift: Expr | float = TIME,
) -> Expr:
    phase = sp.integrate(frequency, TIME).subs(TIME, time_shift)
    return 2 * sp.Abs(2 * (phase % 1) - 1) - 1


def sawtooth(
        frequency: Expr | float | int = Note.A4,
        time_shift: Expr | float = TIME,
) -> Expr:
    phase = sp.integrate(frequency, TIME).subs(TIME, time_shift)
    return 2 * (phase - sp.floor(phase + 0.5))


def sawtooth_v2(
        frequency: Expr | float | int = Note.A4,
        time_shift: Expr | float = TIME,
        resolution: int = 200
) -> Expr:
    i = sp.symbols('i', real=True, positive=True)
    return sp.Sum(((-1)**(i+1) / i) * sine(i*frequency), (i, 1, resolution))


def trigger(
    trigger_time: Expr | float | int = 1.0,
) -> Expr:
    return sp.Piecewise(
        (0.0, trigger_time > TIME),
        (1.0, trigger_time <= TIME)
    )
