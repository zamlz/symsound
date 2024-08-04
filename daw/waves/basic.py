from typing import Union

import sympy as sp
from sympy.core.expr import Expr

from daw.constants import Note
from daw.waves.time import TIME


def sin(
        frequency: Union[Expr, float, int] = Note.A4,
        amplitude: Union[Expr, float, int] = 0.5,
        time: Union[Expr, float, int] = TIME
) -> Expr:
    return amplitude * sp.sin(2 * sp.pi * frequency * time)


def sawtooth(
        frequency: Union[Expr, float, int] = Note.A4,
        amplitude: Union[Expr, float, int] = 0.5,
        time: Union[Expr, float, int] = TIME
) -> Expr:
    return 2 * amplitude * (
        time * frequency - sp.floor(time * frequency + 0.5)
    )
