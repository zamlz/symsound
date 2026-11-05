# Combining signals

Everything in this library is algebra on signals, so a handful of patterns cover
most of what you want to build.

## Mixing

Add signals, and scale each to set its level:

```python
import sympy as sp

from daw.filters import low_pass_filter
from daw.waves.basic import sawtooth, sine, square, trigger
from daw.waves.time import TIME

mix = 0.6 * sine(220) + 0.3 * square(110) + 0.1 * sawtooth(55)
```

The weights should add up to about 1 or less so the mix does not clip.

## Gating: on and off

Multiply by something that is 1 (let through) or 0 (silence):

```python
stutter = (square(8) + 1) / 2          # 1 for half of each cycle, else 0, 8 times a second
gated = sawtooth(220) * stutter
```

`square` runs between -1 and 1, so `square + 1` runs between 0 and 2, and
`(square + 1) / 2` between 0 and 1. `trigger(t)` is a one-way gate.

## A decay that repeats

A sawtooth ramps from -1 to 1 and jumps back. Flipping and squashing it gives a
ramp from 1 down to 0 that restarts every cycle:

```python
ramp = (1 - sawtooth(1)) / 2         # 1 falling to 0 over each second
```

Raising the ramp to a power bends it. A large power makes the sound die away
quickly after each hit, a power below 1 makes it linger and then drop:

```python
fast_decay = ramp ** 20     # a sharp click
slow_decay = ramp ** 2
swell = ramp ** 0.5         # holds up, then falls
```

Multiply this by a tone or noise and you have a repeating drum hit.

## Fading with a time window

`sp.Piecewise` selects different expressions in different time ranges. The first
true condition wins and `True` means "otherwise". Join conditions with
`sp.And`, `sp.Or`, not Python's `and`/`or`:

```python
fade_in = sp.Piecewise(
    (TIME / 2, TIME <= 2),      # rises from 0 to 1 over two seconds
    (1.0, True),                # then stays at 1
)
swelling = fade_in * sine(440)
```

## A bump that rises and falls: `sin²`

`sin(pi * x) ** 2` goes 0, up to 1 at `x = 0.5`, and back to 0 at `x = 1`, with
smooth edges. Put a time window around it and you have a smooth "bump" signal
that is 0 everywhere else. This is the heart of `kick.py`:

```python
start, length = 2.0, 4.0
bump = sp.Piecewise(
    (
        sp.sin(sp.pi * (TIME - start) / length) ** 2,
        sp.And(TIME >= start, TIME <= start + length),
    ),
    (0.0, True),
)
```

`bump` is 0 before 2 s, rises smoothly to 1 at 4 s, falls back to 0 at 6 s, and
stays 0 afterwards.

## Crossfading between a dry and a processed signal

To blend between a signal and a processed copy of it, use a blend amount `wet`
between 0 (all original) and 1 (all processed):

```python
dry = sawtooth(110)
processed = low_pass_filter(dry, 400)
wet = bump                                        # 0 outside the bump, up to 1 inside

result = dry + wet * (processed - dry)
```

When `wet` is 0 the result is exactly `dry`. When `wet` is 1 it is exactly
`processed`. In between it is a proportional mix. Written like this the
original appears once and the processed copy once.

Why blend rather than just switching the filter on? With `wet` at 0 outside the
bump, the sound before and after is *exactly* the original: no filter delay, no
leftover. With the filter always on, even a wide-open cutoff changes the sound a
little.

## Quickly turning a signal on faster than it fades

`sp.Min(1.0, k * bump)` rises `k` times faster than `bump` and then stops at 1.
Use it to make the blend reach full strength early, so the processed sound is
fully there for most of the bump:

```python
quick_wet = sp.Min(1.0, 5 * bump)
```
