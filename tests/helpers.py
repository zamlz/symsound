# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""Independent reference implementations and evaluation helpers for tests"""

import numpy as np
import sympy as sp

from symsound.waves.time import TIME
from symsound.waves.utils import WHITE_NOISE


def evaluate(expr, t, noise=None):
    """Evaluate a signal expression at the given times, always as an array"""
    t = np.asarray(t, dtype=float)
    fn = sp.lambdify((TIME, WHITE_NOISE), expr, "numpy")
    noise = np.zeros_like(t) if noise is None else noise
    return np.broadcast_to(np.asarray(fn(t, noise), dtype=float), t.shape)


def cycle_fraction(frequency, t):
    """Position within a period in [0, 1)"""
    return np.mod(np.asarray(frequency) * np.asarray(t, dtype=float), 1.0)


def ref_square(frequency, t):
    return np.where(cycle_fraction(frequency, t) < 0.5, 1.0, -1.0)


def ref_triangle(frequency, t):
    x = cycle_fraction(frequency, t)
    return np.where(x < 0.5, 1 - 4 * x, 4 * x - 3)


def ref_sawtooth(frequency, t):
    x = cycle_fraction(frequency, t)
    return np.where(x < 0.5, 2 * x, 2 * x - 2)


def away_from_edges(frequency, t, margin=1e-6):
    """True where t is not within margin of a waveform discontinuity or kink"""
    x = cycle_fraction(frequency, t)
    return (np.abs(x) > margin) & (np.abs(x - 0.5) > margin) & (np.abs(x - 1) > margin)


def ref_adsr(attack, decay, sustain, release, active, t):
    """Piecewise definition of an ADSR envelope, written independently of
    the sympy construction in symsound.envelope"""

    def level(time):
        if time <= attack:
            return time / attack if attack > 0 else 1.0
        if time <= attack + decay:
            return 1 + (sustain - 1) * (time - attack) / decay if decay > 0 else sustain
        return sustain

    out = np.empty(len(t))
    for i, time in enumerate(t):
        if time < 0:
            out[i] = 0.0
        elif time <= active:
            out[i] = level(time)
        elif time <= active + release:
            out[i] = level(active) * (active + release - time) / release
        else:
            out[i] = 0.0
    return out
