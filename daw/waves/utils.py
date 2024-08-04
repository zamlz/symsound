from typing import Union

import numpy as np
import sympy as sp
from sympy.core.expr import Expr

from daw.constants import DEFAULT_SAMPLE_RATE, Note
from daw.waves.time import TIME


def compile(
        wave_function: Expr,
        duration: float, 
        sample_rate: int = DEFAULT_SAMPLE_RATE
) -> np.ndarray:
    divisions = int(sample_rate * duration)
    linspace = np.linspace(0, duration, divisions, endpoint=False)
    equation = sp.lambdify(TIME, wave_function, "numpy")
    return equation(linspace)
