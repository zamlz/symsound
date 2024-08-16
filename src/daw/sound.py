import numpy.typing as npt
import sounddevice as sd
import soundfile as sf

from daw.constants import DEFAULT_SAMPLE_RATE


def play(
        compiled_wave: npt.ArrayLike,
        sample_rate: int = DEFAULT_SAMPLE_RATE
) -> None:
    sd.play(compiled_wave, samplerate=sample_rate)
    sd.wait()


def save(
        compiled_wave: npt.ArrayLike,
        output_file: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE
) -> None:
    sf.write(output_file, compiled_wave, sample_rate)
    
