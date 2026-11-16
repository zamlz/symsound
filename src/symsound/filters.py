# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
import numpy as np
import numpy.typing as npt
import sympy as sp
from scipy import signal as scipy_signal
from sympy.core.expr import Expr

from symsound.waves.time import TIME


def apply_low_pass(
    samples: npt.ArrayLike,
    cutoff_frequency: npt.ArrayLike,
    sample_rate: float,
) -> npt.NDArray[np.float64]:
    """First-order (RC) low-pass filter over uniformly spaced samples.

    Integrates dy/dt = 2*pi*fc*(x - y) with the trapezoidal rule, which is
    second order accurate and stable for any cutoff. The filter starts at
    rest, so the first output sample is always zero. The cutoff is either one
    number or one value per sample."""
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    x = np.asarray(samples, dtype=float)
    fc = np.asarray(cutoff_frequency, dtype=float)
    if fc.ndim > 0 and fc.shape != x.shape:
        raise ValueError("cutoff must be a number or have one value per sample")
    if (fc < 0).any():
        raise ValueError("cutoff must not be negative")

    y = np.zeros(len(x))
    if len(x) < 2:
        return y
    if fc.ndim == 0:
        k = np.pi * float(fc) / sample_rate
        b, a = k / (1 + k), (1 - k) / (1 + k)
        y[1:], _ = scipy_signal.lfilter([b, b], [1, -a], x[1:], zi=[b * x[0]])
        return y

    w = 2 * np.pi * fc
    half_step = 0.5 / sample_rate
    prev_x, prev_y, prev_w = float(x[0]), 0.0, float(w[0])
    for n in range(1, len(x)):
        cur_x, cur_w = float(x[n]), float(w[n])
        prev_y = (
            prev_y + half_step * (prev_w * (prev_x - prev_y) + cur_w * cur_x)
        ) / (1 + half_step * cur_w)
        y[n] = prev_y
        prev_x, prev_w = cur_x, cur_w
    return y


class LowPass(sp.Function):
    """Low-pass filtered signal: LowPass(signal, cutoff, TIME).

    It stays an unevaluated expression, so it composes with everything else
    and survives time shifting. When it is compiled, the signal and cutoff
    are evaluated on the sample grid, which arrives as the third argument,
    and filtered there."""

    nargs = 3

    @staticmethod
    def _imp_(
        signal: npt.ArrayLike,
        cutoff_frequency: npt.ArrayLike,
        time: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        t = np.asarray(time, dtype=float)
        if t.ndim != 1:
            raise ValueError("a filtered signal needs a grid of sample times")
        if len(t) < 2:
            return np.zeros(len(t))
        steps = np.diff(t)
        if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-6, atol=0):
            raise ValueError("a filtered signal needs a uniform, increasing grid")
        x = np.broadcast_to(np.asarray(signal, dtype=float), t.shape)
        fc = np.asarray(cutoff_frequency, dtype=float)
        if fc.ndim > 0:
            fc = np.broadcast_to(fc, t.shape)
        return apply_low_pass(x, fc, 1 / steps[0])


def low_pass_filter(
    signal: Expr,
    cutoff_frequency: Expr | float
) -> Expr:
    cutoff = sp.sympify(cutoff_frequency)
    if cutoff.is_negative:
        raise ValueError("cutoff must not be negative")
    return LowPass(sp.sympify(signal), cutoff, TIME)
