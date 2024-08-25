from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import sympy as sp
from sympy.core.expr import Expr

from daw.constants import DEFAULT_SAMPLE_RATE, Note
from daw.waves.time import BPM, TIME


WHITE_NOISE = sp.Symbol('white_noise')


def linspace(duration: float, sample_rate: int) -> npt.ArrayLike:
    divisions = int(sample_rate * duration)
    return np.linspace(0, duration, divisions, endpoint=False)


# FIXME: Is there a better way to create white noise than to just inject
# it at compilation?
def white_noise(duration: float, sample_rate: int) -> npt.ArrayLike:
    num_samples = int(duration * sample_rate)
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
    wave_function: Union[Expr, List[Expr]],
    duration: float = 1.0,
    bpm: Optional[BPM] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    """Plotting utils for signals
    Either an expression or a list of expression can be provided to plot.
    Duration is either interpreted as time in seconds or beats depending on
    whether bpm is provided or not."""
    if isinstance(wave_function, list):
        wave_function_list = wave_function
    else:
        wave_function_list = [wave_function]
    plt.figure(figsize=(10, 4))
    plt.title('Signal Plot')
    plt.xlabel('Time [s]') if bpm is None else plt.xlabel('Beat [b]')
    plt.ylabel('Amplitude')
    duration_in_sec = duration if bpm is None else bpm.time(duration)
    time = linspace(duration_in_sec, sample_rate)
    x_axis = time if bpm is None else np.vectorize(bpm.beat)(time)
    for wave_func in wave_function_list:
        signal = compile(wave_func, duration_in_sec, sample_rate)
        plt.plot(x_axis, signal)
        plt.grid(True)
    plt.show()  
