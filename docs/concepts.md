# Concepts

## A signal is a formula

In this library a sound is not a list of samples. It is a mathematical
expression of one symbol, `TIME`, the time in seconds:

```python
from daw.waves.basic import sine
from daw.waves.time import TIME

tone = sine(440)
print(tone)
```

`sine(440)` is the expression `sin(880.0*pi*time)`: 440 cycles per second. Nothing has been computed yet.
Because it is an expression you can do algebra with it:

```python
quieter = 0.5 * tone              # half the volume
two_tones = sine(440) + sine(660)  # both at once
tremolo = sine(440) * (0.6 + 0.4 * sine(5))  # volume wobbles 5 times a second
```

Adding signals mixes them. Multiplying a signal by a number changes its volume.
Multiplying two signals shapes one with the other, which is how envelopes and
low-frequency oscillators (LFOs) work.

## Amplitude

Signals are expected to stay between -1 and 1. Anything beyond that clips when
played or saved. Mixing several loud signals can add up past 1, so scale the
sum down, for example `0.3 * (a + b + c)`.

## `TIME`

`TIME` is one shared symbol, `daw.waves.time.TIME`. Every function in the
library builds its expressions from it, so they all line up on the same clock.
Time zero is the start of the audio.

You can use it directly, for example a signal that rises linearly for a second:

```python
ramp = TIME            # 0 at the start, 1 at one second, 2 at two seconds...
```

## From expression to sound

An expression only becomes audio when you ask for it:

```python
from daw.waves.utils import compile

samples = compile(tone, duration=1.0)   # a NumPy array
print(len(samples))                     # 44100 samples: one second at 44.1 kHz
```

`compile` evaluates the expression at 44100 evenly spaced times per second
(`DEFAULT_SAMPLE_RATE`), starting at time zero. `play`, `save` and `plot` call
it for you. See [Playing, saving and plotting](output.md).

## Noise

`daw.waves.utils.WHITE_NOISE` is a second symbol. When compiled it becomes a
fresh random value in [-1, 1] for every sample:

```python
from daw.waves.utils import WHITE_NOISE

hiss = 0.1 * WHITE_NOISE
```

Every use of `WHITE_NOISE` inside one expression sees the *same* random
samples. So `WHITE_NOISE - WHITE_NOISE` is exactly zero, not two independent
noises.

## Piecewise expressions

Anything that changes behaviour at a certain time is written with SymPy's
`Piecewise`. It reads like an if/elif list: the first condition that is true
wins.

```python
import sympy as sp

# silent until one second, then a tone
late_tone = sp.Piecewise(
    (0.0, TIME < 1),
    (sine(440), True),       # True means "otherwise"
)
```

The library's envelopes, `trigger` and the sweep in `kick.py` are all built
this way. See [Combining signals](combining.md).
