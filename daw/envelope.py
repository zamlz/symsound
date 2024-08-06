from typing import Union

import sympy as sp

from sympy.core.expr import Expr
from daw.waves.time import TIME


def adsr_envelope(
        attack: float,
        decay: float,
        sustain: float,
        release: float,
        active: float = 100000
) -> Expr:
    attack_func = TIME / attack

    decay_to_sustain_slope = (sustain - 1.0) / decay
    decay_func = decay_to_sustain_slope * (TIME - attack) + 1

    attack_to_sustain_func = sp.Piecewise(
        (attack_func, TIME <= attack),
        (decay_func, sp.And(TIME > attack, TIME <= (attack + decay))),
        (sustain, TIME > (attack + decay))
    )

    env_at_release = attack_to_sustain_func.subs(TIME, active)
    release_slope = -1 * env_at_release / release
    release_func = release_slope * (TIME - active - release)

    env_piecewise = sp.Piecewise(
        (attack_to_sustain_func, TIME <= active),
        (release_func, sp.And(TIME > active, TIME <= (active + release))),
        (0.0, TIME > (active + release))
    )
    return env_piecewise
