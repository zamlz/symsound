from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import sympy as sp
from sympy.core.expr import Expr

from daw.constants import DEFAULT_SAMPLE_RATE, Note
from daw.waves.time import TIME


WHITE_NOISE = sp.Symbol('white_noise')


def linspace(duration: float, sample_rate: int) -> npt.ArrayLike:
    divisions = int(sample_rate * duration)
    return np.linspace(0, duration, divisions, endpoint=False)


# FIXME: Is there a better way to create white noise than to just inject
# it at compilation?
def white_noise(duration: float, sample_rate: int) -> npt.ArrayLike:
    num_samples = duration * sample_rate
    return np.random.uniform(-0.5, 0.5, num_samples)


def compile(
        wave_function: Expr,
        duration: float, 
        sample_rate: int = DEFAULT_SAMPLE_RATE
) -> npt.ArrayLike:
    equation = sp.lambdify((TIME, WHITE_NOISE), wave_function, "numpy")
    return equation(
        linspace(duration, sample_rate),
        white_noise(duration, sample_rate)
    )


def plot(
    wave_functions: Union[Expr, List[Expr]],
    duration: float = 1,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> None:
    if not isinstance(wave_functions, list):
        wave_functions = [wave_functions]
    plt.figure(figsize=(10, 4))
    plt.title('Signal Plot')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    for wave_function in wave_functions:
        time = linspace(duration, sample_rate)
        signal = compile(wave_function, duration, sample_rate)
        plt.plot(time, signal)
        plt.grid(True)
    plt.show()  
