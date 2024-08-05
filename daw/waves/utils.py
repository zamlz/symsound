from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.core.expr import Expr

from daw.constants import DEFAULT_SAMPLE_RATE, Note
from daw.waves.time import TIME


def linspace(duration: float, sample_rate: int) -> np.ndarray:
    divisions = int(sample_rate * duration)
    return np.linspace(0, duration, divisions, endpoint=False)


def compile(
        wave_function: Expr,
        duration: float, 
        sample_rate: int = DEFAULT_SAMPLE_RATE
) -> np.ndarray:
    equation = sp.lambdify(TIME, wave_function, "numpy")
    return equation(linspace(duration, sample_rate))


def plot_signal(
    wave_function: Expr,
    duration: float = 1,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> None:
    time = linspace(duration, sample_rate)
    signal = compile(wave_function, duration, sample_rate)
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal)
    plt.title('Signal Plot')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()  
