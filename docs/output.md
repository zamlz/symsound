# Playing, saving and plotting

All in `daw.waves.utils`. Each takes a signal expression and a length.

```python
from daw.waves.basic import sine
from daw.waves.utils import compile, plot, save

tone = sine(440)
samples = compile(tone, duration=0.5)                 # NumPy array, 22050 samples
save(tone, duration=1.0, output_file="tone.wav")      # a mono WAV file
```

| Function | What it does |
| --- | --- |
| `compile(expr, duration, sample_rate=44100)` | Returns the samples as a NumPy array |
| `play(expr, duration=1.0, sample_rate=44100)` | Plays through the speakers and waits until it finishes |
| `save(expr, duration, output_file, sample_rate=44100)` | Writes a WAV file |
| `plot(expr_or_list, duration=1.0, bpm=None)` | Shows the waveform in a matplotlib window |

`duration` is in seconds. `play` needs a working audio device; the others do
not. Press Ctrl-C to stop `play` early.

```python
from daw.waves.utils import play

play(tone, 2.0)          # needs speakers
```

## Plotting in beats

Pass a `BPM` and the duration is read as beats, with a beats axis:

```python
from daw.waves.time import BPM

plot([sine(2), sine(3)], duration=4, bpm=BPM(120))   # four beats, two signals
```

Plotting is for looking at the shape of a signal. Keep the duration short and
the frequencies low or the picture is just a solid block.

## Length matters

Your expression is evaluated from time zero for exactly `duration` seconds. Make
the duration at least as long as the sounds you want to hear, including their
release tails. A song's length is the end of its last note plus that note's
release.

## Helpers

`daw.waves.utils.linspace(duration, sample_rate)` makes the array of sample times
that `compile` uses, and `white_noise(duration, sample_rate)` makes the noise
samples that replace `WHITE_NOISE`. You rarely need them directly.
