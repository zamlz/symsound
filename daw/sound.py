import numpy as np
import sounddevice as sd
import soundfile as sf
import sympy as sp

from daw.constants import DEFAULT_SAMPLE_RATE
from daw.waves import AbstractWaveFunction


def play_audio(
        compiled_wave: np.ndarray,
        sample_rate=DEFAULT_SAMPLE_RATE
) -> None:
    sd.play(compiled_wave, samplerate=sample_rate)
    sd.wait()


def save_audio(
        compiled_wave: np.ndarray,
        output_file: str,
        sample_rate=DEFAULT_SAMPLE_RATE
) -> None:
    sf.write(output_file, compiled_wave, sample_rate)
    
