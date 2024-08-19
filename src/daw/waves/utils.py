from typing import Union

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
