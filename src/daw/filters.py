from typing import Union

import sympy as sp
from sympy.core.expr import Expr

from daw.constants import Note
from daw.waves.time import TIME


def low_pass_filter(
    signal: Expr,
    cutoff_frequency: Expr | float
) -> Expr:
    s = sp.symbols('s')
    laplace_signal = sp.laplace_transform(signal, TIME, s, noconds=True)
    tau = 1 / (2 * sp.pi * cutoff_frequency)
    filter = 1 / (tau * s + 1)
    filtered_laplace_signal = laplace_signal * filter
    return sp.inverse_laplace_transform(filtered_laplace_signal, s, TIME)
