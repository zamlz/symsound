# Oscillators

An oscillator is a repeating wave. All of them live in `daw.waves.basic` and
all take the same two arguments:

- `frequency`: how many cycles per second, as a number or an expression.
- `time_shift`: advanced use, see [Time and rhythm](time-and-rhythm.md).

```python
from daw.constants import Note
from daw.waves.basic import sawtooth, sine, square, triangle

a = sine(440)
b = square(220)
c = triangle(110)
d = sawtooth(55)
```

| Function | Shape | Sound |
| --- | --- | --- |
| `sine` | smooth curve | pure, soft |
| `square` | jumps between +1 and -1 | hollow, buzzy |
| `triangle` | straight lines up and down | between sine and square |
| `sawtooth` | ramps up, jumps down | bright, harsh |

All four run between -1 and 1. `sawtooth_v2` is an additive approximation that
overshoots (see [Gotchas](gotchas.md)); prefer `sawtooth`.

## Notes

`daw.constants.Note` is a table of equal-tempered pitches in Hz, from `A0`
(27.5 Hz) to `C8`. Sharps are written `AS4` (A sharp 4), `FS4` and so on.
Notes are plain floats, so you can do arithmetic with them:

```python
middle_c = sine(Note.C4)
fifth_above = sine(Note.C4 * 1.5)    # about a perfect fifth up
octave_up = sine(Note.C4 * 2)
```

## Frequency can be an expression

Pass a signal as the frequency and the pitch changes over time. This is how
vibrato and frequency modulation (FM) work:

```python
vibrato = sine(440 + 10 * sine(5))      # 440 Hz, wobbling +-10 Hz five times a second
siren = sine(440 + 200 * sine(0.5))     # slow sweep between 240 and 640 Hz
```

The oscillator integrates the frequency to get the phase, so the pitch really
follows the expression. Keep the frequency positive.

## Trigger

`trigger(t)` is 0 before time `t` and 1 from `t` onward. It is a switch, useful
for starting or stopping things:

```python
from daw.waves.basic import trigger

starts_at_two_seconds = trigger(2.0) * sine(440)
stops_at_two_seconds = (1 - trigger(2.0)) * sine(440)
```
