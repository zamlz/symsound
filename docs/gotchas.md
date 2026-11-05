# Gotchas

Things that are surprising, or known not to work, before they cost you an
afternoon.

## Clipping

Signals above 1 or below -1 are clipped when played or saved. Mixed signals add
up: scale the mix. `examples/kick.py` and `examples/kick_with_low_pass.py` peak at about 1.19 and clip a little.

## Time shifting a plain oscillator does not silence the start

`sine(440).subs(TIME, TIME - 2)` is a sine delayed by 2 s, but it is still
sounding before 2 s (it is a sine at negative time). Multiply by `trigger(2)` or
an [ADSR envelope](envelopes.md) to make it start silent.

## `time_shift` and `subs` move in opposite directions

`osc(f, time_shift=TIME + x)` starts the oscillator `x` seconds *into* its cycle.
`expr.subs(TIME, TIME - x)` starts the whole expression `x` seconds *later*. See
[Time and rhythm](time-and-rhythm.md).

## Use `sp.And` / `sp.Or`, not `and` / `or`

Python's `and` and `or` do not work on SymPy conditions. Write
`sp.And(TIME >= a, TIME <= b)`. Also write comparisons with `TIME` on the left.

## Noise is shared inside one expression

Every `WHITE_NOISE` in an expression sees the same samples, so
`WHITE_NOISE - WHITE_NOISE` is exactly zero. Each separate `compile`
call, though, draws new noise, so two compiles of a noisy signal differ.

## Filters need a grid of times

A filtered signal can be compiled, played, saved or plotted, but not evaluated
at a single time. The filter always starts at rest at time zero. See
[Filters](filters.md).

## Constant expressions compile to a single number

`compile(sp.Float(0.5), 1.0)` returns a single value, not 44100 samples, so
`play` and `save` of a pure constant (silence or a DC offset) do not work. This
is a known bug. Anything that still depends on `TIME` or `WHITE_NOISE`
compiles fine.

## `sawtooth_v2` overshoots

`sawtooth_v2` builds the wave from 200 sine harmonics and overshoots 1 by about
18%. Use `sawtooth` unless you specifically want the band-limited shape.
