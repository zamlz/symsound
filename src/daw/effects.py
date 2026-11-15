# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""Time effects built from one idea: a signal, later.

`delay(signal, seconds)` is the signal as it was `seconds` ago, and silence
before that. Everything else here is a sum of delays: an echo is a few copies
of a sound at even spacings, a reverb is thousands at irregular ones (see
docs/effects.md for both, built from `delay`).
"""

import sympy as sp
from sympy.core.expr import Expr

from daw.filters import LowPass
from daw.waves.time import TIME


def _shift(expression: Expr, seconds: Expr) -> Expr:
    """The expression with `TIME` replaced by `TIME - seconds`, right through
    any filter in it.

    A filter starts at rest on the first sample, so shifting its time argument
    would leave it already warmed up by whatever the signal did before time
    zero. Shifting a filter's output later is the same as filtering a later
    input, so the input is shifted and silenced before `seconds`, and the
    filter starts at rest where its signal starts."""
    if not expression.has(LowPass):
        return expression.subs(TIME, TIME - seconds)
    if isinstance(expression, LowPass):
        signal, cutoff, _ = expression.args
        started = sp.Piecewise((1, TIME >= seconds), (0, True))
        return LowPass(
            started * _shift(signal, seconds), _shift(cutoff, seconds), TIME
        )
    return expression.func(*[_shift(arg, seconds) for arg in expression.args])


def delay(signal: Expr, seconds: Expr | float) -> Expr:
    """The signal `seconds` later, silent until then.

    The result is an ordinary expression of `TIME`, so it can be mixed, scaled,
    shifted, filtered and delayed again like any other. The delay must be a
    plain number of seconds that is not negative."""
    amount = sp.sympify(seconds)
    if not amount.is_number or not amount.is_real:
        raise ValueError("delay must be a plain number of seconds")
    if amount.is_negative:
        raise ValueError("delay must not be negative")
    signal = sp.sympify(signal)
    if amount.is_zero:
        return signal
    return sp.Piecewise((_shift(signal, amount), TIME >= amount), (0, True))
