
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import soundfile as sf
import sympy as sp
from matplotlib.figure import Figure
from scipy import signal as scipy_signal
from sympy.core.expr import Expr

from daw.constants import DEFAULT_SAMPLE_RATE
from daw.waves.time import BPM, TIME

WHITE_NOISE = sp.symbols('white_noise', real=True)


def linspace(duration: float, sample_rate: int) -> npt.NDArray[np.float64]:
    divisions = int(sample_rate * duration)
    return np.linspace(0, duration, divisions, endpoint=False)


# FIXME: Is there a better way to create white noise than to just inject
# it at compilation?
def white_noise(duration: float, sample_rate: int) -> npt.NDArray[np.float64]:
    num_samples = int(duration * sample_rate)
    return np.random.uniform(-1.0, 1.0, num_samples)


def compile(
        wave_function: Expr,
        duration: float,
        sample_rate: int = DEFAULT_SAMPLE_RATE
) -> npt.NDArray[np.float64]:
    equation = sp.lambdify((TIME, WHITE_NOISE), wave_function, "numpy")
    times = linspace(duration, sample_rate)
    samples = equation(times, white_noise(duration, sample_rate))
    # A constant expression (silence, a DC offset) comes back as one bare
    # number: repeat it so there is always one value per sample
    return np.array(np.broadcast_to(np.asarray(samples, dtype=float), times.shape))


def play(
    wave_function: Expr,
    duration: float = 1.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> None:
    # Imported lazily so the module works without an audio backend
    import sounddevice as sd

    compiled_wave = compile(wave_function, duration, sample_rate)
    try:
        sd.play(compiled_wave, samplerate=sample_rate)
        sd.wait()
    except KeyboardInterrupt:
        pass


def save(
    wave_function: Expr,
    duration: float,
    output_file: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> None:
    compiled_wave = compile(wave_function, duration, sample_rate)
    sf.write(output_file, compiled_wave, sample_rate)


def _window(
    wave_function: Expr,
    start: float,
    duration: float,
    sample_rate: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Times and samples of [start, start + duration).

    The signal is compiled from time zero and then cut, so anything with a
    state (like a filter) is where it would really be at `start`."""
    total = start + duration
    times = np.asarray(linspace(total, sample_rate), dtype=float)
    samples = compile(wave_function, total, sample_rate)
    first = int(round(start * sample_rate))
    return times[first:], samples[first:]


def _finish(fig: Figure, output_file: str | os.PathLike[str] | None) -> None:
    if output_file is None:
        plt.show()
        return
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot(
    wave_function: Expr | list[Expr],
    duration: float = 1.0,
    bpm: BPM | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    start: float = 0.0,
    labels: list[str] | None = None,
    title: str = 'Signal Plot',
    stacked: bool = False,
    output_file: str | os.PathLike[str] | None = None,
) -> None:
    """Plotting utils for signals
    Either an expression or a list of expression can be provided to plot.
    Duration is either interpreted as time in seconds or beats depending on
    whether bpm is provided or not, and so is start, the time the plot
    begins at (the signal is still computed from time zero).
    With labels, every signal is named: in a legend, or as a title above
    its own panel when stacked is true. Stacked draws one panel per
    signal, sharing the time axis. With output_file the figure is written
    to that file (an image format, e.g. png) instead of being shown."""
    waves = wave_function if isinstance(wave_function, list) else [wave_function]
    if not waves:
        raise ValueError('plot needs at least one signal')
    if labels is not None and len(labels) != len(waves):
        raise ValueError('labels needs one name per signal')
    if start < 0:
        raise ValueError('start must not be negative')

    def seconds(amount: float) -> float:
        return amount if bpm is None else bpm.time(amount)

    if stacked:
        fig, axes = plt.subplots(
            len(waves), 1, sharex=True, squeeze=False,
            figsize=(10, 1.6 * len(waves) + 1),
        )
        panels = list(axes[:, 0])
    else:
        fig = plt.figure(figsize=(10, 4))
        panels = [fig.gca()] * len(waves)
    for index, (wave, ax) in enumerate(zip(waves, panels, strict=True)):
        times, samples = _window(
            wave, seconds(start), seconds(duration), sample_rate
        )
        x_axis = times if bpm is None else np.vectorize(bpm.beat)(times)
        name = labels[index] if labels is not None else None
        ax.plot(x_axis, samples, label=name)
        ax.grid(True)
        if stacked:
            ax.set_ylabel('Amplitude')
            if name is not None:
                ax.set_title(name, loc='left', fontsize=10)
    last = panels[-1]
    last.set_xlabel('Time [s]' if bpm is None else 'Beat [b]')
    panels[0].set_title(title)
    if not stacked:
        last.set_ylabel('Amplitude')
        if labels is not None:
            last.legend()
    _finish(fig, output_file)


SPECTROGRAM_WINDOW = 2048  # samples per analysis window
SPECTROGRAM_RANGE_DB = 90


def spectrogram(
    wave_function: Expr,
    duration: float = 1.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    start: float = 0.0,
    max_frequency: float | None = None,
    title: str = 'Spectrogram',
    output_file: str | os.PathLike[str] | None = None,
) -> None:
    """Show how the frequency content of a signal changes over time: time
    across, frequency up, loudness as colour (a 90 dB range ending at the
    loudest point). Time is in seconds. Start and duration pick a window,
    though the signal is still computed from time zero. max_frequency limits
    the vertical axis (default: half the sample rate). With output_file the
    figure is written to that file instead of being shown."""
    if duration <= 0:
        raise ValueError('duration must be positive')
    if start < 0:
        raise ValueError('start must not be negative')
    _, samples = _window(wave_function, start, duration, sample_rate)
    if len(samples) < SPECTROGRAM_WINDOW:
        raise ValueError(
            f'window too short to analyse: need at least {SPECTROGRAM_WINDOW} '
            f'samples, got {len(samples)}'
        )
    _, _, power = scipy_signal.spectrogram(
        samples,
        fs=sample_rate,
        nperseg=SPECTROGRAM_WINDOW,
        noverlap=3 * SPECTROGRAM_WINDOW // 4,
    )
    decibels = 10 * np.log10(power + 1e-20)
    loudest = decibels.max()

    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(
        decibels,
        origin='lower',
        aspect='auto',
        extent=(start, start + len(samples) / sample_rate, 0, sample_rate / 2),
        vmin=loudest - SPECTROGRAM_RANGE_DB,
        vmax=loudest,
        cmap='magma',
    )
    fig.colorbar(image, ax=ax, label='Power [dB]')
    ax.set_ylim(0, max_frequency if max_frequency is not None else sample_rate / 2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title(title)
    _finish(fig, output_file)
