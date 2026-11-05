# Walkthrough: `examples/kick.py`

`kick.py` makes a 30 second beat: a kick drum, a hi-hat and a stuttering saw
buzz, with a low-pass filter sweep in the middle third. This page goes through
it a piece at a time. The code blocks are the real file, in order.

First, the imports and the tempo.

```python
import sympy as sp

from daw.filters import low_pass_filter
from daw.waves.basic import sawtooth, sine, square
from daw.waves.time import BPM, TIME
from daw.waves.utils import WHITE_NOISE, compile, play, plot, save

bpm = BPM(140)
```

`sp` is SymPy, needed for `Piecewise`, `sin`, `exp` and friends. `bpm` converts
beats to seconds: at 140 BPM one beat is 0.43 s. See [Time and
rhythm](time-and-rhythm.md).

## The repeating decay (the trick behind every drum here)

All three drums use the same idea, from [Combining signals](combining.md):
`(1 - sawtooth(f)) / 2` is a ramp that falls from 1 to 0 and restarts every
cycle, and raising it to a power bends it into a decay shape.

```python
kick_frequency = bpm.frequency(1)
kick_time_shift = TIME + bpm.time(0.5)
kick_beat = (1 - sawtooth(kick_frequency, time_shift=kick_time_shift)) / 2
kick_decay = .5

kick = (kick_beat**kick_decay) * sine(40)
```

- `bpm.frequency(1)` is one cycle per beat, so the ramp restarts every beat.
- `time_shift=TIME + bpm.time(0.5)` offsets the ramp by half a cycle so that
  it starts at its peak (1) exactly at time zero and falls to 0 by the end of
  the beat. Without the shift the ramp would start halfway down.
- `kick_beat ** 0.5` (a square root) keeps the level high for longer and then
  drops it at the end, a thumpy shape.
- `* sine(40)` is the body of the drum: a 40 Hz tone, loudest at the start of each
  beat and fading away before the next one.

So `kick` is a 40 Hz thump on every beat.

```python
hat_frequency = bpm.frequency(1)
hat_time_shift = TIME + bpm.time(1)
hat_beat = (1 - sawtooth(hat_frequency, time_shift=hat_time_shift)) / 2
hat_decay = 20

hat = 0.25 * (hat_beat**hat_decay) * WHITE_NOISE
```

The hat is the same recipe with three differences:

- It is noise (`WHITE_NOISE`), not a tone, so it sounds like a hiss.
- The decay power is 20, so each hit is a very short tick.
- Its shift is a whole beat, which on its own would change nothing. But
  the kick shifts by half a beat, so the hat's peak lands *half a beat later*
  than the kick's: on the off-beat, between kicks.
- `0.25 *` keeps it quieter than the kick.

```python
saw_frequency = bpm.frequency(4)
saw_time_shift = TIME + bpm.time(4)
saw_beat = (1 - sawtooth(saw_frequency, time_shift=saw_time_shift)) / 2
saw_decay = 9

stutter_lfo = 6*sine(bpm.frequency(4)) + 12.5
stutter = square(stutter_lfo) + 1.0

saw = 0.1 * (saw_beat**saw_decay) * sawtooth(1000) * stutter
```

The saw hit is a buzzy 1000 Hz sawtooth that fires once per four-beat bar
(`bpm.frequency(4)`), at the middle of the bar, with a medium decay (power 9).

The `stutter` chops the buzz on and off like a gate:

- `stutter_lfo = 6*sine(bpm.frequency(4)) + 12.5` is a number that slowly
  moves between 6.5 and 18.5 over each bar.
- `square(stutter_lfo)` is a square wave whose *frequency is that number*, so it
  flips between +1 and -1 between about 6 and 18 times a second, getting faster
  and slower through the bar. This is frequency modulation: see
  [Oscillators](oscillators.md).
- `+ 1.0` turns the -1/+1 into 0/2: off or on (at double volume).

Multiplying the buzz by `stutter` gates it on and off at that varying rate.
`0.1 *` keeps it quiet.

```python
dry = kick + hat + saw
```

`dry` is the whole beat with no effects: the three drums added together
("dry" is the audio term for unprocessed).

## The filter sweep

The goal: for the first 10 seconds play `dry` as is, for the next 10 seconds
sweep a low-pass filter down and back up, and for the last 10 seconds play
`dry` again, exactly as before.

```python
duration = 30
third = duration / 3
```

`third` is 10 seconds: the length of each section.

```python
sweep = sp.Piecewise(
    (
        sp.sin(sp.pi * (TIME - third) / third) ** 2,
        sp.And(TIME >= third, TIME <= 2 * third),
    ),
    (0.0, True),
)
```

`sweep` is a signal that says "how far through the sweep are we", from 0 to 1:

- Between 10 s and 20 s (`TIME >= third and TIME <= 2 * third`) it is
  `sin(pi * u) ** 2`, where `u = (TIME - third) / third` runs from 0 to 1 over
  those 10 seconds. That curve starts at 0, climbs smoothly to 1 at 15 s, and
  comes back to 0 at 20 s.
- Everywhere else (`(0.0, True)`: "otherwise") it is 0.

You can picture it like this:

```text
sweep
1 |             ___
  |           /     \
  |         /         \
0 |________/           \________
  0       10    15    20       30  seconds
```

Everything else in the sweep is derived from this one curve.

```python
cutoff = 12000 * sp.exp(-sp.log(12000 / 150) * sweep)
```

`cutoff` is the filter's cutoff frequency in Hz, driven by `sweep`. This line is
the one that looks strangest, so step by step:

- When `sweep` is 0, `exp(-... * 0)` is 1 and the cutoff is `12000`: wide open,
  hardly any filtering.
- When `sweep` is 1, `exp(-log(12000 / 150))` is `150 / 12000`, so the cutoff
  is `12000 * 150 / 12000 = 150`: very closed, only bass gets through.
- In between it slides along a curve that is exponential, not a straight line.
  That is intentional: we hear frequencies as ratios (an octave is a doubling),
  so an even-sounding sweep multiplies by the same factor each moment instead of
  subtracting the same number of Hz.

| time | `sweep` | `cutoff` |
| --- | --- | --- |
| 0 to 10 s | 0 | 12000 Hz |
| 12.5 s | 0.5 | about 1340 Hz |
| 15 s | 1 | 150 Hz |
| 17.5 s | 0.5 | about 1340 Hz |
| 20 to 30 s | 0 | 12000 Hz |

```python
wet = sp.Min(1.0, 5 * sweep)
```

`wet` is how much of the filtered sound to use: 0 means none, 1 means all of it.
`5 * sweep` rises five times as fast as `sweep`, and `Min(1.0, ...)` stops it at
1. So `wet` is 0 until the sweep starts, rises to 1 about a second and a half
into it, stays at 1, and falls back to 0 at the end.

Why not use the filter on its own? Even with the cutoff at 12 kHz, a filter
still changes the sound slightly. Blending it in from exactly 0 guarantees the first and last
thirds are the original sound with nothing left over.

```python
w = dry + wet * (low_pass_filter(dry, cutoff) - dry)
```

This is the crossfade from [Combining signals](combining.md):

- `low_pass_filter(dry, cutoff)` is the beat run through the filter, with the
  cutoff sweeping as described.
- `wet * (filtered - dry)` is the *difference* the filter makes, scaled by `wet`.
- Adding that to `dry` gives `dry` when `wet` is 0, the filtered sound when `wet`
  is 1, and a mix in between.

`w` is the finished piece.

```python
play(w, duration)
save(w, duration, "beat.wav")
```

`play` sends it to the speakers and waits; `save` writes `beat.wav`. Both
compile the expression to 30 seconds of samples first. See
[Playing, saving and plotting](output.md).

## Changing it

| To change | Edit |
| --- | --- |
| Tempo | `BPM(140)` |
| How hard the kick thumps | `kick_decay` (smaller is longer), the `40` in `sine(40)` for its pitch |
| How snappy the hat is | `hat_decay` (bigger is shorter) |
| Total length, and each third | `duration` |
| How closed the filter gets | the `150` in `cutoff` (lowest cutoff) |
| How open it starts | the two `12000`s in `cutoff` |
| How fast the filter fades in | the `5` in `wet` (bigger is quicker) |
| Make it a different effect | swap `low_pass_filter` for any signal you want to fade in |
