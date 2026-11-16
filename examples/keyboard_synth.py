#!/usr/bin/env python3

import sympy as sp

from symsound.envelope import adsr_envelope
from symsound.keyboard import keyboard
from symsound.waves.basic import sine
from symsound.waves.time import TIME

# A glassy electric piano you play on the computer keyboard (a s d f ... for the
# white keys, w e t ... for the black ones, z and x for the octave).
#
# Three layers, all plain algebra of TIME, so it works one note at a time:
#
#   bell   FM: a sine whose phase is bent by a second sine one octave up. The
#          bend (the modulation index) starts large and dies away, so every
#          note begins bright and clangy and mellows into a pure tone.
#   sub    a sine an octave below, for weight.
#   drive  the sum is squashed with tanh, which adds warm odd harmonics and
#          keeps every note inside +/-1, however hard it is driven.
#
# On top of that a tremolo (loudness wobble) fades in after the attack, and an
# ADSR envelope shapes the note. The envelope is what tells the keyboard when
# the key came up: without it the note could not be released.

bell_ratio = 2           # the modulator is an octave above the carrier
bell_brightness = 3.0    # how bent the phase starts out
bell_fade = 4.0          # per second: how fast the brightness dies away
bell_floor = 0.35        # the little bit of bend that is left

sub_level = 0.5
drive = 1.6

tremolo_rate = 5.5       # Hz
tremolo_depth = 0.18
tremolo_fade_in = 1.0    # seconds until the wobble is at full depth

envelope = adsr_envelope(attack=0.004, decay=0.7, sustain=0.3, release=0.6)


def synth(note):
    """The sound of one key: a signal in TIME, and in ACTIVE_DURATION through
    the envelope"""
    index = bell_floor + bell_brightness * sp.exp(-bell_fade * TIME)
    carrier = 2 * sp.pi * note * TIME
    bell = sp.sin(carrier + index * sp.sin(bell_ratio * carrier))

    voice = bell + sub_level * sine(note / 2)
    driven = sp.tanh(drive * voice)

    swell = sp.Min(1, TIME / tremolo_fade_in)
    tremolo = 1 - tremolo_depth * swell * (1 - sine(tremolo_rate)) / 2

    return driven * tremolo * envelope


# Only open the keyboard when run as a script, so the sound above can be
# imported (the tests do this to measure it)
if __name__ == "__main__":
    keyboard(synth, title="keyboard_synth.py")
