# Time and rhythm

## BPM

`BPM` converts between beats and seconds so you can write music in beats:

```python
from daw.waves.time import BPM

bpm = BPM(140)
print(bpm.time(1))        # seconds in one beat: 0.4285...
print(bpm.time(4))        # seconds in a bar of four beats
print(bpm.frequency(1))   # a signal that repeats once per beat: 2.333 Hz
print(bpm.frequency(4))   # repeats once per four beats
print(bpm.beat(0.4285714285714286))   # which beat it is at that many seconds: 1.0
```

`bpm.frequency(n)` is the one to remember. A wave with that frequency completes
one cycle every `n` beats, which locks it to the tempo. `sine(bpm.frequency(1))`
is an LFO that goes round once per beat.

## Two ways to shift a signal in time

### 1. Shifting the whole expression: `subs(TIME, TIME - delay)`

To start a signal `delay` seconds later, replace `TIME` with `TIME - delay`:

```python
import sympy as sp
from daw.waves.basic import sine, trigger
from daw.waves.time import TIME

note = sine(440)
later = note.subs(TIME, TIME - 2.0)    # the same note, two seconds later
```

This is how songs are built: each note is made once, then shifted to where it
belongs. `examples/ode_to_joy.py` does exactly this for every note.

Careful: an expression that is non-zero before it starts, like a plain sine, is
still non-zero before the shift. Delaying it just moves the waveform; it does
not silence the part before. To start it silent, multiply by an envelope or by
`trigger`:

```python
starts_silent = sine(440).subs(TIME, TIME - 2.0) * trigger(2.0)
```

[ADSR envelopes](envelopes.md) are silent before time 0 by design, so a note
with an envelope can be shifted freely.

### 2. The `time_shift` argument on oscillators

Each oscillator takes `time_shift`, an expression to use in place of `TIME`
inside that one oscillator. `TIME + x` starts the oscillator `x` seconds *into*
its cycle, which is the opposite direction from `subs`:

```python
from daw.waves.basic import sawtooth

normal = sawtooth(1)
ahead = sawtooth(1, time_shift=TIME + 0.25)   # starts a quarter cycle in
```

Use it to offset the phase of a repeating wave so its peak lands where you
want it relative to the beat. `kick.py` uses this to place the kick, hat and
saw hits within the bar. A shift by a whole number of cycles changes nothing.

## Placing notes in a song

The usual pattern is a loop that keeps a running position in beats:

```python
from daw.constants import Note
from daw.envelope import ACTIVE_DURATION, adsr_envelope
from daw.waves.time import TIME

bpm = BPM(120)
melody = [(Note.C4, 1), (Note.E4, 1), (None, 1), (Note.G4, 2)]   # (pitch, beats)

song = 0
position = 0   # in beats
for pitch, beats in melody:
    if pitch is not None:      # None is a rest: only move the position along
        note = sine(pitch) * adsr_envelope(0.05, 0.1, 0.6, 0.1)
        note = note.subs(ACTIVE_DURATION, bpm.time(beats))   # how long it is held
        note = note.subs(TIME, TIME - bpm.time(position))    # where it starts
        song = song + note
    position += beats
```

Compile `song` for `bpm.time(position)` seconds to hear the whole phrase.
