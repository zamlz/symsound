"""Effects that place a sound between the ears.

These turn a mono signal into a `StereoExpr` (see `daw.stereo`). Like every
effect here they are plain expressions of `TIME`.
"""

import sympy as sp
from sympy.core.expr import Expr

from daw.stereo import StereoExpr
from daw.waves.time import TIME


def pan(signal: Expr, position: Expr | float) -> StereoExpr:
    """A mono signal placed between the ears.

    `position` runs from -1 (all left) through 0 (centre) to 1 (all right). The
    gains follow an equal-power law, `cos` for the left and `sin` for the right
    as the position moves across a quarter turn, so `left**2 + right**2` is the
    same everywhere and the sound does not get louder or quieter as it moves.
    At the centre each side is `1/sqrt(2)`.

    The position is either a plain number, which must be between -1 and 1, or an
    expression of `TIME` that moves the sound as it plays (`sine(0.5)` sweeps it
    from ear to ear and back every two seconds). A moving position is held
    between -1 and 1, so a wave that overshoots stops at the edge instead of
    flipping the sound out of phase."""
    place = sp.sympify(position)
    if place.free_symbols - {TIME}:
        raise ValueError("pan position may only depend on TIME")
    if place.free_symbols:
        place = sp.Min(1, sp.Max(-1, place))
    elif not place.is_real:
        raise ValueError("pan position must be a real number or a function of TIME")
    elif place < -1 or place > 1:
        raise ValueError("pan position must be between -1 and 1")
    signal = sp.sympify(signal)
    angle = sp.pi * (place + 1) / 4
    return StereoExpr(sp.cos(angle) * signal, sp.sin(angle) * signal)
