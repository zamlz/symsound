# Envelopes

A raw oscillator is a constant drone. An envelope shapes its loudness over the
life of a note, from `daw.envelope`:

```python
from daw.envelope import ACTIVE_DURATION, adsr_envelope
from daw.waves.basic import sine

env = adsr_envelope(attack=0.1, decay=0.2, sustain=0.5, release=0.3)
```

An ADSR envelope has four stages, all in seconds except sustain:

```text
level
1 |      /\
  |     /  \
  |    /    \_____________      <- sustain level (0 to 1)
  |   /                   \
0 |__/                     \____
     A    D       S         R
   attack decay           release
```

- **attack**: rises from 0 to 1.
- **decay**: falls from 1 to the sustain level.
- **sustain**: a level (0 to 1), held while the note is held.
- **release**: falls from wherever the envelope was to 0 once the note is released.

A duration of `0` skips that stage (an attack of 0 jumps straight to full volume).

## How long is the note held?

The envelope does not know how long you hold the note. That is the symbol
`ACTIVE_DURATION`, which you fill in with `subs`:

```python
note = sine(440) * env.subs(ACTIVE_DURATION, 1.0)    # held for one second
```

The note sounds for the held time plus the release. The envelope is exactly
zero before time 0 and after the release has finished.

If the note is released early, in the middle of the attack or decay, the release
starts from whatever level it had reached.

## Several notes with different lengths

Build the envelope once and fill in a different `ACTIVE_DURATION` for each use:

```python
env = adsr_envelope(0.01, 0.1, 0.5, 0.2)
short = sine(440) * env.subs(ACTIVE_DURATION, 0.25)
long = sine(550) * env.subs(ACTIVE_DURATION, 2.0)
```

To place each note in a song, see [Time and rhythm](time-and-rhythm.md).

## Percussive decays without ADSR

Drum hits do not need ADSR. A repeating ramp raised to a power makes a decay
that restarts every cycle. See the kick, hat and saw in
[the `kick.py` walkthrough](kick-walkthrough.md).
