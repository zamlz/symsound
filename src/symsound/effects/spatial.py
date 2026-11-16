"""Effects that place a sound between the ears.

These turn a mono signal into a `StereoExpr` (see `symsound.stereo`). Like every
effect here they are plain expressions of `TIME`.
"""

import sympy as sp
from sympy.core.expr import Expr

from symsound.stereo import StereoExpr
from symsound.waves.time import TIME


def pan(signal: Expr, angle: Expr | float) -> StereoExpr:
    """A mono signal placed between the ears by an angle, in radians.

    The angle is measured from the centre: 0 is the middle, `pi/4` (45 degrees)
    is all right and `-pi/4` is all left. The gains are the sine and cosine of
    the angle from hard left, `left = cos(pi/4 + angle)` and
    `right = sin(pi/4 + angle)`, so `left**2 + right**2` is the same for every
    angle and the sound never gets louder or quieter as it moves. At the centre
    each side is `1/sqrt(2)`.

    The angle is any number, or any expression of `TIME` that moves the sound
    as it plays (`pi/4 * sine(0.5)` sweeps it from ear to ear and back every two
    seconds). It is not limited: the gains are periodic, one turn is `2*pi`, and
    past `pi/4` the sound keeps turning past the edge, one side changing sign
    (a copy in anti-phase) until it comes round again. For a sound that stays
    between the ears, keep the angle within `pi/4` of the centre. Write `pi` as
    `sympy.pi` for exact gains: the far ear is then exactly silent at the edge,
    where a float like `numpy.pi / 4` leaves a trace of rounding error."""
    turn = sp.sympify(angle)
    if turn.free_symbols - {TIME}:
        raise ValueError("pan angle may only depend on TIME")
    if not turn.free_symbols and not turn.is_real:
        raise ValueError("pan angle must be a real number or a function of TIME")
    signal = sp.sympify(signal)
    from_left = sp.pi / 4 + turn
    return StereoExpr(sp.cos(from_left) * signal, sp.sin(from_left) * signal)
