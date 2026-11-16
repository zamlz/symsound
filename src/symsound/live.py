# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""Real-time rendering of notes: the engine behind the live tools.

`compile` turns a whole expression into samples at once. A keyboard cannot
wait for that: notes start when a key goes down and end when it comes up, and
the sound card asks for a few hundred samples at a time. So the work is split.

* `NoteSource` does the slow, symbolic part once (`sympy.lambdify`) and then
  renders any stretch of a note as plain NumPy, in a few microseconds.
* `Mixer` keeps the notes that are sounding and sums them, one chunk at a time.
* `Instrument` makes a `NoteSource` for each note of a function `f(note)`.
* `Player` plays a finished signal, with pause and seek.
* `RingBuffer` remembers the latest samples so a display can draw them.
* `LiveOutput` connects a `Mixer` to the speakers.
"""

import threading
from collections.abc import Callable, Hashable, Iterable
from types import TracebackType
from typing import Protocol

import numpy as np
import numpy.typing as npt
import sympy as sp
from sympy.core.expr import Expr

from symsound.constants import DEFAULT_SAMPLE_RATE, Note
from symsound.envelope import ACTIVE_DURATION
from symsound.filters import LowPass
from symsound.waves.time import TIME
from symsound.waves.utils import WHITE_NOISE

# While a key is down the note has no end yet. `active_duration` is set to
# this, which is far later than any note, so an envelope stays in its sustain
RUNNING = 1e9

# An expression with no envelope would stop with a click when its key comes up:
# fade it out over this many seconds instead
FADE_OUT = 0.01

# A released note that never falls to exact silence is cut off after this long
LONGEST_RELEASE = 10.0

Samples = npt.NDArray[np.float64]


class NoteSource:
    """A signal compiled for rendering in pieces.

    Time is counted from the moment the note starts, so `render(0, n)` is
    the first `n` samples of the note whenever it is played. The expression may
    use `TIME`, `WHITE_NOISE` and the envelope's `ACTIVE_DURATION`.

    A filtered expression is refused: a filter remembers its past, so it
    cannot render an arbitrary stretch of the signal."""

    def __init__(
        self, expression: Expr, sample_rate: int = DEFAULT_SAMPLE_RATE
    ) -> None:
        expression = sp.sympify(expression)
        if expression.has(LowPass):
            raise ValueError(
                "a filtered signal cannot be rendered live: "
                "a filter needs the whole signal from the start"
            )
        if sample_rate <= 0:
            raise ValueError("sample rate must be positive")
        self.sample_rate = sample_rate
        self.has_release = bool(expression.has(ACTIVE_DURATION))
        self._function = sp.lambdify(
            (TIME, WHITE_NOISE, ACTIVE_DURATION), expression, "numpy"
        )

    def render(
        self, first: int, count: int, active_duration: float = RUNNING
    ) -> Samples:
        """`count` samples of the note, starting at sample number `first`.

        `active_duration` is how long the key was held, in seconds, which is
        what an envelope's release is measured from."""
        if first < 0 or count < 0:
            raise ValueError("first and count must not be negative")
        times = (first + np.arange(count)) / self.sample_rate
        noise = np.random.uniform(-1.0, 1.0, count)
        samples = self._function(times, noise, active_duration)
        # A constant expression comes back as one bare number
        return np.array(np.broadcast_to(np.asarray(samples, dtype=float), times.shape))


class _Voice:
    """One note that is sounding: where it is up to, and whether it was released"""

    def __init__(self, key: Hashable, source: NoteSource) -> None:
        self.key = key
        self.source = source
        self.position = 0  # samples rendered so far
        self.released_at: int | None = None
        self.finished = False

    def release(self) -> None:
        if self.released_at is None:
            self.released_at = self.position

    def render(self, count: int) -> Samples:
        rate = self.source.sample_rate
        if self.released_at is None:
            samples = self.source.render(self.position, count)
        elif self.source.has_release:
            samples = self.source.render(
                self.position, count, self.released_at / rate
            )
            since = self.position + count - self.released_at
            silent = count > 0 and not samples.any()
            if silent or since >= LONGEST_RELEASE * rate:
                self.finished = True
        else:
            samples = self.source.render(self.position, count)
            fade = max(int(FADE_OUT * rate), 1)
            gone = self.position + np.arange(count) - self.released_at
            samples = samples * np.clip(1 - gone / fade, 0.0, 1.0)
            if self.position + count - self.released_at >= fade:
                self.finished = True
        self.position += count
        return samples


class Mixer:
    """The notes that are sounding, summed.

    `note_on(key, source)` starts a note and `note_off(key)` releases it. The
    key is anything hashable that names the finger on the keyboard (a `Note`,
    a key code). Pressing a key that is already down does nothing, which makes
    key auto-repeat harmless; pressing one that is still dying away starts a
    new note and lets the old one finish.

    `render(count)` returns the next `count` samples of the mix. Calls may
    come from another thread (the audio callback), so they are locked. Presses
    and releases take effect at the start of the next chunk, so their timing
    is only as exact as the chunk size.

    `gain` scales the sum, because notes add up and a chord would clip. The
    default of 0.5 keeps two full-scale notes under 1; use 1 / n for n notes."""

    def __init__(self, gain: float = 0.5) -> None:
        if gain < 0:
            raise ValueError("gain must not be negative")
        self.gain = gain
        self._voices: list[_Voice] = []
        self._lock = threading.Lock()

    def note_on(self, key: Hashable, source: NoteSource) -> None:
        with self._lock:
            if any(v.key == key and v.released_at is None for v in self._voices):
                return
            self._voices.append(_Voice(key, source))

    def note_off(self, key: Hashable) -> None:
        with self._lock:
            for voice in self._voices:
                if voice.key == key:
                    voice.release()

    def is_held(self, key: Hashable) -> bool:
        with self._lock:
            return any(v.key == key and v.released_at is None for v in self._voices)

    @property
    def active(self) -> int:
        """How many notes are sounding, including ones still dying away"""
        with self._lock:
            return len(self._voices)

    def render(self, count: int) -> Samples:
        if count < 0:
            raise ValueError("count must not be negative")
        mix = np.zeros(count)
        with self._lock:
            for voice in self._voices:
                mix += voice.render(count)
            self._voices = [v for v in self._voices if not v.finished]
        return self.gain * mix


class Player:
    """A finished signal that can be played, paused and sought, a chunk at a time.

    It is what `visualize` plays: `render(count)` hands out the next `count`
    samples while it is playing (silence when paused) and moves the position
    on, so the audio callback drives the clock and a display that reads
    `position` is always in step with what is heard. Playback stops by itself
    at the end; `play` from the end starts again. Calls may come from
    different threads, so they are locked."""

    def __init__(self, samples: npt.ArrayLike, sample_rate: int) -> None:
        data = np.array(samples, dtype=float)
        if data.ndim != 1 or len(data) == 0:
            raise ValueError("samples must be a non-empty one-dimensional array")
        if sample_rate <= 0:
            raise ValueError("sample rate must be positive")
        self.samples = data
        self.sample_rate = sample_rate
        self._position = 0
        self._playing = False
        self._lock = threading.Lock()

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def position(self) -> int:
        """The next sample to be played"""
        with self._lock:
            return self._position

    @property
    def seconds(self) -> float:
        return self.position / self.sample_rate

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._playing

    @property
    def finished(self) -> bool:
        return self.position >= len(self.samples)

    def play(self) -> None:
        with self._lock:
            if self._position >= len(self.samples):
                self._position = 0
            self._playing = True

    def pause(self) -> None:
        with self._lock:
            self._playing = False

    def toggle(self) -> None:
        if self.playing:
            self.pause()
        else:
            self.play()

    def seek(self, seconds: float) -> None:
        """Jump to a time, kept between the start and the end"""
        target = round(seconds * self.sample_rate)
        with self._lock:
            self._position = min(max(target, 0), len(self.samples))

    def replay(self) -> None:
        self.seek(0.0)
        self.play()

    def render(self, count: int) -> Samples:
        if count < 0:
            raise ValueError("count must not be negative")
        out = np.zeros(count)
        with self._lock:
            if not self._playing:
                return out
            chunk = self.samples[self._position : self._position + count]
            out[: len(chunk)] = chunk
            self._position += len(chunk)
            if self._position >= len(self.samples):
                self._playing = False
        return out


class Instrument:
    """A function from a note to its signal, compiled once per note.

    `Instrument(f)` makes `f(Note.A4)` a `NoteSource` the first time it is
    asked for and remembers it. `precompile` does that for many notes up front,
    which is the slow part, so a keyboard
    does it before the first key goes down and never while you play."""

    def __init__(
        self,
        function: Callable[[Note], Expr],
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.function = function
        self.sample_rate = sample_rate
        self._sources: dict[Note, NoteSource] = {}

    def source(self, note: Note) -> NoteSource:
        if note not in self._sources:
            self._sources[note] = NoteSource(self.function(note), self.sample_rate)
        return self._sources[note]

    def precompile(self, notes: Iterable[Note]) -> None:
        for note in notes:
            self.source(note)


class RingBuffer:
    """The most recent `capacity` samples, for a display to draw.

    The audio thread writes and the display thread reads, so both are locked.
    `latest(count)` returns the newest `count` samples, oldest first, with
    zeros in front if fewer than that have been written so far."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data = np.zeros(capacity)
        self._written = 0
        self._lock = threading.Lock()

    @property
    def written(self) -> int:
        """Samples written since the start: tells a display how much is new"""
        with self._lock:
            return self._written

    def write(self, samples: npt.ArrayLike) -> None:
        data = np.asarray(samples, dtype=float)
        chunk = data[-self.capacity :]
        with self._lock:
            self._written += len(data)
            end = self._written % self.capacity
            # The newest sample lands just before `end`
            indexes = np.arange(end - len(chunk), end) % self.capacity
            self._data[indexes] = chunk

    def latest(self, count: int) -> Samples:
        if not 0 <= count <= self.capacity:
            raise ValueError("count must be between 0 and the capacity")
        with self._lock:
            end = self._written % self.capacity
            indexes = np.arange(end - count, end) % self.capacity
            out = self._data[indexes].copy()
            missing = max(count - self._written, 0)
        out[:missing] = 0.0
        return out


class Renderer(Protocol):
    """Anything that can hand the sound card its next samples: a `Mixer`, a `Player`"""

    def render(self, count: int) -> Samples: ...


class LiveOutput:
    """Plays a `Mixer` (or a `Player`) through the speakers until stopped.

    The sound card asks for `block_size` samples at a time; each request is
    answered by `mixer.render`, and copied into `tap` (a `RingBuffer`) when
    one is given. Use it as a context manager, or call `start` and `stop`.
    `sounddevice` is only imported when this is created."""

    def __init__(
        self,
        mixer: Renderer,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = 512,
        tap: RingBuffer | None = None,
    ) -> None:
        import sounddevice as sd

        self.mixer = mixer
        self.tap = tap
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            blocksize=block_size,
            channels=1,
            dtype="float32",
            callback=self._fill,
        )

    def _fill(
        self,
        outdata: npt.NDArray[np.float32],
        frames: int,
        time: object,
        status: object,
    ) -> None:
        samples = self.mixer.render(frames)
        outdata[:, 0] = samples
        if self.tap is not None:
            self.tap.write(samples)

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()

    def __enter__(self) -> LiveOutput:
        self.start()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.stop()
