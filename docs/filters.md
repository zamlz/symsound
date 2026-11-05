# Filters

`daw.filters.low_pass_filter(signal, cutoff)` removes the high-frequency part of
a sound and keeps the low part. The higher a frequency is above the cutoff, the
more it is turned down. It is the "muffled" sound: a hand over a speaker, music
heard through a wall.

```python
from daw.filters import low_pass_filter
from daw.waves.basic import sawtooth, sine

bright = sawtooth(110)
dull = low_pass_filter(bright, 500)    # keep what is below roughly 500 Hz
```

The result is an ordinary signal expression. Mix it, scale it, shift it or
filter it again like any other.

## What the cutoff means

The cutoff is the frequency where the filter has turned the sound down to about
70% (-3 dB). It is a gentle, first-order filter:

| Frequency | Level after the filter |
| --- | --- |
| well below the cutoff | unchanged |
| at the cutoff | 0.71 of the original |
| 10x the cutoff | about 0.1 of the original |
| 100x the cutoff | about 0.01 |

That is -20 dB per decade, a gradual roll-off rather than a wall. Filtering
twice makes it steeper:

```python
steeper = low_pass_filter(low_pass_filter(bright, 500), 500)
```

A cutoff of `0` blocks everything. Cutoffs must not be negative.

## Filter anything

The filter works on any signal: oscillators, noise, envelopes, sums, shifted
notes.

```python
from daw.waves.utils import WHITE_NOISE

rumble = low_pass_filter(WHITE_NOISE, 200)      # noise with the treble removed
muffled_chord = low_pass_filter(sine(220) + sine(277) + sine(330), 300)
```

## A cutoff that changes over time

The cutoff may itself be a signal, so the filter can sweep. The cutoff
expression must stay above zero:

```python
wah = low_pass_filter(sawtooth(110), 800 + 600 * sine(2))   # between 200 and 1400 Hz, twice a second
```

For a smooth sweep that sounds even, move the cutoff exponentially, not
linearly, because we hear pitch as ratios. Moving from 12 kHz to 150 Hz is
`12000 * exp(-log(12000 / 150) * s)`, where `s` goes from 0 to 1. That is the
trick in the [`kick.py` walkthrough](kick-walkthrough.md).

## How it behaves

- The filter starts at rest at time zero. The first sample is silent and the
  output then follows the input.
- It adds a small delay (phase lag) that grows with frequency. You will not hear
  it on its own, but it matters if you mix filtered and unfiltered copies of the
  same sound. See [Combining signals](combining.md).
- A filtered expression can only be evaluated over a whole run of evenly spaced
  sample times, which is what `compile`, `play` and `save` do. It cannot be
  evaluated at a single time.

## Under the hood

`low_pass_filter` returns a `LowPass` expression, a placeholder that holds the
signal and the cutoff. When the expression is compiled, the signal and cutoff are
evaluated at every sample time and handed to `apply_low_pass(samples, cutoff,
sample_rate)`, which does the filtering on the sample array. You can call
`apply_low_pass` directly on a NumPy array if you already have samples; the
cutoff is one number or one value per sample.
