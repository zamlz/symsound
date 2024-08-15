from typing import Union

import sympy as sp

from sympy.core.expr import Expr
from daw.waves.time import TIME


def adsr_envelope(
        attack: float,
        decay: float,
        sustain: float,
        release: float,
        active: float = 100000,
) -> Expr:
    if attack < 0 or decay < 0 or sustain < 0 or release < 0 or active < 0:
        raise Exception("Invalid paramter value (less than zero)")

    if attack > 0.0:
        attack_func = TIME / attack
    else:
        attack_func = 1.0

    if decay > 0.0:
        decay_to_sustain_slope = (sustain - 1.0) / decay
        decay_func = decay_to_sustain_slope * (TIME - attack) + 1
    else:
        decay_func = sustain

    attack_to_sustain_func = sp.Piecewise(
        (attack_func, TIME <= attack),
        (decay_func, sp.And(TIME > attack, TIME <= (attack + decay))),
        (sustain, TIME > (attack + decay))
    )

    if release > 0.0:
        env_at_release = attack_to_sustain_func.subs(TIME, active)
        release_slope = -1 * env_at_release / release
        release_func = release_slope * (TIME - active - release)
    else:
        release_func = 0.0

    env_piecewise = sp.Piecewise(
        (attack_to_sustain_func, TIME <= active),
        (release_func, sp.And(TIME > active, TIME <= (active + release))),
        (0.0, TIME > (active + release))
    )
    return env_piecewise
