# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""StereoExpr: two signals, one for each ear.

A mono signal is one expression of `TIME`. A stereo signal is a `StereoExpr`, a pair
of such expressions: `StereoExpr(left, right)`. Nothing else changes. Each side is an
ordinary expression, so everything that works on a signal (mixing, scaling,
`delay`, `low_pass_filter`, envelopes) works on a side, and `StereoExpr.map` applies
one function to both sides at once. `pan`, which places a mono signal between
the ears, is an effect and lives in `symsound.effects`. See docs/stereo.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import sympy as sp
from sympy.core.expr import Expr


@dataclass(frozen=True)
class StereoExpr:
    """A left and a right signal, played together.

    Both are sympified, so numbers are accepted. Compiling a `StereoExpr` gives an
    array of shape (samples, 2), left in column 0."""

    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        object.__setattr__(self, "left", sp.sympify(self.left))
        object.__setattr__(self, "right", sp.sympify(self.right))

    def map(self, function: Callable[[Expr], Expr]) -> StereoExpr:
        """The same function applied to the left and to the right"""
        return StereoExpr(function(self.left), function(self.right))


def mono_to_stereo(signal: Expr) -> StereoExpr:
    """A mono signal heard equally in both ears"""
    return StereoExpr(signal, signal)
