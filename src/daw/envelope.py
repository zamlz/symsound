# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham

import sympy as sp
from sympy.core.expr import Expr

from daw.waves.time import TIME

ACTIVE_DURATION = sp.symbols('active_duration')


def adsr_envelope(
        attack: float,
        decay: float,
        sustain: float,
        release: float,
) -> Expr:
    if attack < 0 or decay < 0 or sustain < 0 or release < 0:
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
        (0.0, TIME < 0),
        (attack_func, sp.And(TIME >= 0, TIME <= attack)),
        (decay_func, sp.And(TIME > attack, TIME <= (attack + decay))),
        (sustain, TIME > (attack + decay))
    )

    if release > 0.0:
        env_at_release = attack_to_sustain_func.subs(TIME, ACTIVE_DURATION)
        release_slope = -1 * env_at_release / release
        release_func = release_slope * (TIME - ACTIVE_DURATION - release)
    else:
        release_func = 0.0

    env_piecewise = sp.Piecewise(
        (attack_to_sustain_func, TIME <= ACTIVE_DURATION),
        (
            release_func,
            sp.And(TIME > ACTIVE_DURATION, TIME <= (ACTIVE_DURATION + release)),
        ),
        (0.0, TIME > (ACTIVE_DURATION + release))
    )
    return env_piecewise
