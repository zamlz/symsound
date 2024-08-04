import numpy as np
import sounddevice as sd
import soundfile as sf
import sympy as sp

from daw.constants import DEFAULT_SAMPLE_RATE, time


def _linspace(duration: float, sample_rate: int) -> np.ndarray:
    divisions = int(sample_rate * duration)
    return np.linspace(0, duration, divisions, endpoint=False)


def compile_wave(
        wave,
        duration: float,
        sample_rate=DEFAULT_SAMPLE_RATE
) -> None:
    np_wave_function = sp.lambdify(time, wave, "numpy")
    return np_wave_function(_linspace(duration, sample_rate))


def play_wave(
        wave,
        duration: float,
        sample_rate=DEFAULT_SAMPLE_RATE
) -> None:
    compiled_wave = compile_wave(wave, duration, sample_rate)
    sd.play(compiled_wave, samplerate=sample_rate)
    sd.wait()


def save_compiled_wave(
        compiled_wave: np.ndarray,
        output_file: str,
        sample_rate=DEFAULT_SAMPLE_RATE
) -> None:
    sf.write(output_file, compiled_wave, sample_rate)
    
