#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""A short sine pulse with an echo, whose delay you choose when it starts.

Run it and type the number of seconds between the echoes (or press Enter for
the default). Every echo is the pulse again, delayed with `delay` and half as
loud as the one before it.
"""

import math

import sympy as sp

from daw.effects import delay
from daw.envelope import ACTIVE_DURATION, adsr_envelope
from daw.waves.basic import sine
from daw.waves.time import TIME
from daw.waves.utils import play

pulse_frequency = 440
pulse_length = 0.1  # seconds the pulse is held, before its short fade
pulse_gain = 0.5  # keeps even badly overlapping echoes under 1

default_delay = 0.25  # seconds between echoes
feedback = 0.5  # each echo is this much as loud as the one before
repeats = 8
tail = 0.5  # seconds of quiet left after the last echo

# A sine that comes in and goes out smoothly, so the pulse has no clicks
envelope = adsr_envelope(attack=0.005, decay=0.0, sustain=1.0, release=0.03)
held = envelope.subs(ACTIVE_DURATION, pulse_length)
pulse = pulse_gain * sine(pulse_frequency) * held


def echo(signal, spacing):
    """The signal, then `repeats` copies of it `spacing` seconds apart"""
    return signal + sum(
        feedback**k * delay(signal, k * spacing) for k in range(1, repeats + 1)
    )


def ask_for_delay(default=default_delay):
    """Ask how many seconds apart the echoes are, until the answer makes sense"""
    while True:
        try:
            answer = input(f"Seconds between echoes [{default}]: ").strip()
        except EOFError:  # nobody to ask (input is piped): use the default
            return default
        if not answer:
            return default
        try:
            seconds = float(answer)
        except ValueError:
            print(f"{answer!r} is not a number")
            continue
        if math.isfinite(seconds) and seconds > 0:
            return seconds
        print("the delay must be more than zero seconds")


# Only play when run as a script, so importing this file is silent
if __name__ == "__main__":
    spacing = ask_for_delay()
    duration = pulse_length + repeats * spacing + tail
    print(f"{repeats} echoes, {spacing} s apart: playing for {duration:.1f} s")
    play(echo(pulse, spacing), duration=duration)
