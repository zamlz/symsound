"""Echo and reverb: effects that put a sound's own past back on top of it.

Like a filter, an effect remembers what came before, so it is a placeholder
expression (`Echo`, `Reverb`) that does its work when the signal is compiled
and the whole run of samples is known.

* `echo` repeats the sound after a delay, each repeat quieter than the last.
* `reverb` adds the dense, fading wash of a room (Schroeder's design: eight
  parallel comb filters into four all-pass filters, with the delays of Jezar's
  Freeverb).
"""

import numpy as np
import numpy.typing as npt
import sympy as sp
from scipy import signal as scipy_signal
from sympy.core.expr import Expr

from daw.waves.time import TIME

# Delays of the reverb's comb and all-pass filters, in samples at 44100 Hz
# (Freeverb's tuning). They share no common factor, so their echoes do not
# line up and the tail sounds smooth rather than ringing at one pitch.
COMB_DELAYS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
ALLPASS_DELAYS = (556, 441, 341, 225)
ALLPASS_GAIN = 0.5
TUNING_RATE = 44100

# The impulse response is kept until it has fallen 90 dB: 1.5 times the
# time it takes to fall 60 dB
TAIL_LENGTH = 1.5


def _grid_rate(time: npt.ArrayLike) -> float | None:
    """Sample rate of a grid of sample times, None if it is too short to say"""
    t = np.asarray(time, dtype=float)
    if t.ndim != 1:
        raise ValueError("an effect needs a grid of sample times")
    if len(t) < 2:
        return None
    steps = np.diff(t)
    if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-6, atol=0):
        raise ValueError("an effect needs a uniform, increasing grid")
    return float(1 / steps[0])


def _check_rate(sample_rate: float) -> None:
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")


def _check_level(level: float) -> None:
    if level < 0:
        raise ValueError("level must not be negative")


def apply_echo(
    samples: npt.ArrayLike,
    delay: float,
    feedback: float,
    level: float,
    sample_rate: float,
) -> npt.NDArray[np.float64]:
    """Echo over uniformly spaced samples.

    The first repeat comes `delay` seconds later at `level` times the height
    of the sound, and every repeat after it is `feedback` times the one before.
    The delay is rounded to a whole number of samples, and the output is as
    long as the input, so repeats that fall past the end are cut off."""
    _check_rate(sample_rate)
    _check_level(level)
    if not 0 <= feedback < 1:
        raise ValueError("feedback must be at least 0 and less than 1")
    shift = round(delay * sample_rate)
    if shift < 1:
        raise ValueError("delay must be at least one sample")
    x = np.asarray(samples, dtype=float)
    # y[n] = x[n] + level * sum over k >= 1 of feedback**(k-1) * x[n - k*shift]
    numerator = np.zeros(shift + 1)
    numerator[0] = 1.0
    numerator[shift] = level - feedback
    denominator = np.zeros(shift + 1)
    denominator[0] = 1.0
    denominator[shift] = -feedback
    return np.asarray(scipy_signal.lfilter(numerator, denominator, x))


def reverb_impulse_response(
    decay: float, damping: float, sample_rate: float
) -> npt.NDArray[np.float64]:
    """What the reverb adds to a single click: the room's sound, with energy 1.

    The tail falls by 60 dB in `decay` seconds (at low pitches; `damping`, from
    0 to just under 1, makes the highs die away faster, like soft walls). The
    energy is scaled to 1 so that `level` means the same thing for any room:
    white noise put through the wet part comes out at the height it went in."""
    _check_rate(sample_rate)
    if decay <= 0:
        raise ValueError("decay must be positive")
    if not 0 <= damping < 1:
        raise ValueError("damping must be at least 0 and less than 1")
    length = int(TAIL_LENGTH * decay * sample_rate) + 4096
    click = np.zeros(length)
    click[0] = 1.0

    def delayed(tuned: int) -> int:
        return max(round(tuned * sample_rate / TUNING_RATE), 1)

    combs = np.zeros(length)
    for tuned in COMB_DELAYS:
        shift = delayed(tuned)
        # Falls 60 dB (a factor 1000) every `decay` seconds
        gain = 10 ** (-3 * shift / (decay * sample_rate))
        numerator = np.zeros(shift + 2)
        numerator[shift] = 1.0
        numerator[shift + 1] = -damping
        denominator = np.zeros(shift + 2)
        denominator[0] = 1.0
        denominator[1] -= damping
        denominator[shift] -= gain * (1 - damping)
        combs += scipy_signal.lfilter(numerator, denominator, click)

    response = combs
    for tuned in ALLPASS_DELAYS:
        shift = delayed(tuned)
        numerator = np.zeros(shift + 1)
        numerator[0] = -ALLPASS_GAIN
        numerator[shift] = 1.0
        denominator = np.zeros(shift + 1)
        denominator[0] = 1.0
        denominator[shift] = -ALLPASS_GAIN
        response = scipy_signal.lfilter(numerator, denominator, response)
    return np.asarray(response / np.sqrt(np.sum(response**2)))


def apply_reverb(
    samples: npt.ArrayLike,
    decay: float,
    level: float,
    damping: float,
    sample_rate: float,
) -> npt.NDArray[np.float64]:
    """Reverb over uniformly spaced samples: the sound plus `level` times its
    reverberation (see `reverb_impulse_response`). The output is as long as the
    input, so a tail that falls past the end is cut off."""
    _check_level(level)
    x = np.asarray(samples, dtype=float)
    response = reverb_impulse_response(decay, damping, sample_rate)
    wet = scipy_signal.fftconvolve(x, response)[: len(x)]
    return np.asarray(x + level * wet)


def _on_grid(signal: npt.ArrayLike, time: npt.ArrayLike) -> npt.NDArray[np.float64]:
    t = np.asarray(time, dtype=float)
    return np.array(np.broadcast_to(np.asarray(signal, dtype=float), t.shape))


class Echo(sp.Function):
    """Echoed signal: Echo(signal, delay, feedback, level, TIME).

    It stays an unevaluated expression, so it composes with everything else.
    When it is compiled, the signal is evaluated on the sample grid, which
    arrives as the last argument, and the echoes are added there."""

    nargs = 5

    @staticmethod
    def _imp_(
        signal: npt.ArrayLike,
        delay: float,
        feedback: float,
        level: float,
        time: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        rate = _grid_rate(time)
        x = _on_grid(signal, time)
        if rate is None:
            return x
        return apply_echo(x, delay, feedback, level, rate)


class Reverb(sp.Function):
    """Reverberated signal: Reverb(signal, decay, level, damping, TIME).

    It stays an unevaluated expression, so it composes with everything else.
    When it is compiled, the signal is evaluated on the sample grid, which
    arrives as the last argument, and the reverb is added there."""

    nargs = 5

    @staticmethod
    def _imp_(
        signal: npt.ArrayLike,
        decay: float,
        level: float,
        damping: float,
        time: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        rate = _grid_rate(time)
        x = _on_grid(signal, time)
        if rate is None:
            return x
        return apply_reverb(x, decay, level, damping, rate)


def _number(name: str, value: Expr | float) -> sp.Expr:
    """A setting that fixes the shape of the effect, so it cannot vary with time"""
    number = sp.sympify(value)
    if not number.is_number or not number.is_real:
        raise ValueError(f"{name} must be a plain number")
    return sp.Float(float(number))


def echo(
    signal: Expr,
    delay: Expr | float = 0.3,
    feedback: Expr | float = 0.5,
    level: Expr | float = 0.5,
) -> Expr:
    """Repeat a sound after `delay` seconds, again and again, each repeat
    `feedback` times as loud as the one before (less than 1, or it would never
    die away). The first repeat is `level` times as loud as the sound itself."""
    settings = (
        _number("delay", delay),
        _number("feedback", feedback),
        _number("level", level),
    )
    if settings[0] <= 0:
        raise ValueError("delay must be positive")
    if not 0 <= settings[1] < 1:
        raise ValueError("feedback must be at least 0 and less than 1")
    if settings[2] < 0:
        raise ValueError("level must not be negative")
    return Echo(sp.sympify(signal), *settings, TIME)


def reverb(
    signal: Expr,
    decay: Expr | float = 1.5,
    level: Expr | float = 0.3,
    damping: Expr | float = 0.5,
) -> Expr:
    """Put a sound in a room. The reverb dies away over `decay` seconds (the
    time to fall 60 dB), `level` is how loud it is next to the dry sound, and
    `damping`, from 0 to just under 1, makes the high frequencies fade first."""
    settings = (
        _number("decay", decay),
        _number("level", level),
        _number("damping", damping),
    )
    if settings[0] <= 0:
        raise ValueError("decay must be positive")
    if settings[1] < 0:
        raise ValueError("level must not be negative")
    if not 0 <= settings[2] < 1:
        raise ValueError("damping must be at least 0 and less than 1")
    return Reverb(sp.sympify(signal), *settings, TIME)
