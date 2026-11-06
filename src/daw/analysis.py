"""Looking at a signal: the numbers behind the spectrogram and the overview.

These are plain NumPy, with no window or toolkit, so the same code draws a
finished recording and a live keyboard, and can be tested exactly.
"""

from typing import Protocol

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

WINDOW = 2048  # samples analysed for each column
HOP = 512  # samples between one column and the next
LOWEST_FREQUENCY = 30.0  # Hz at the bottom row

Samples = npt.NDArray[np.float64]

# Columns are analysed this many at a time, to keep a long signal's memory small
_BATCH = 512


def frequency_of_row(
    row: float, rows: int, sample_rate: int, low: float = LOWEST_FREQUENCY
) -> float:
    """The frequency in Hz at a spectrogram row, counted up from the bottom.

    The rows are spaced evenly in pitch, from `low` to just under half the sample
    rate, so each octave gets the same height."""
    if rows < 2:
        raise ValueError("a spectrogram needs at least two rows")
    high = sample_rate / 2
    return float(low * (high / low) ** (row / (rows - 1)))


def row_of_frequency(
    frequency: float, rows: int, sample_rate: int, low: float = LOWEST_FREQUENCY
) -> float:
    """The row (counted up from the bottom) where a frequency is drawn"""
    if rows < 2:
        raise ValueError("a spectrogram needs at least two rows")
    high = sample_rate / 2
    return float((rows - 1) * np.log(frequency / low) / np.log(high / low))


def spectrogram_columns(
    samples: npt.ArrayLike,
    sample_rate: int,
    rows: int = 256,
    *,
    window: int = WINDOW,
    hop: int = HOP,
    range_db: float = 80.0,
) -> Samples:
    """The spectrogram of a signal as an array of shape (columns, rows).

    Column `k` analyses the `window` samples that end at sample `(k + 1) * hop`,
    so the column for the playhead at sample `p` is `p // hop`. Rows go up in
    pitch (see `frequency_of_row`). Values run from 0 to 1: 0 is `range_db` below
    a full-scale sine, 1 is a full-scale sine or louder. The scale is fixed, not
    relative to the loudest part, so a quiet passage looks quiet."""
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    if window < 2 or hop < 1 or rows < 2 or range_db <= 0 or sample_rate <= 0:
        raise ValueError("window, hop, rows, range and sample rate must be positive")
    if len(x) == 0:
        return np.zeros((0, rows))

    # A row is wider than a frequency bin near the top of the scale and
    # narrower near the bottom. It shows the loudest bin it covers, or, when it
    # covers none, the level interpolated between the two bins around it
    bin_width = sample_rate / window
    last_bin = window // 2
    centre = np.array([frequency_of_row(r, rows, sample_rate) for r in range(rows)])
    edges = np.sqrt(centre[:-1] * centre[1:])  # halfway, in pitch, between rows
    edges = np.concatenate([[centre[0] ** 2 / edges[0]], edges, [sample_rate / 2]])
    first = np.clip(np.ceil(edges[:-1] / bin_width), 0, last_bin).astype(int)
    stop = np.clip(np.floor(edges[1:] / bin_width), 0, last_bin).astype(int) + 1
    position = np.clip(centre / bin_width, 0, last_bin)
    lower = np.minimum(position.astype(int), last_bin - 1)
    weight = position - lower

    padded = np.concatenate([np.zeros(window - hop), x])
    needed = (len(x) + hop - 1) // hop * hop + window - hop
    padded = np.concatenate([padded, np.zeros(needed - len(padded))])
    frames = sliding_window_view(padded, window)[::hop]
    taper = np.hanning(window)
    scale = 2.0 / taper.sum()  # a full-scale sine then has magnitude 1

    out = np.empty((len(frames), rows))
    for start in range(0, len(frames), _BATCH):
        batch = frames[start : start + _BATCH] * taper
        magnitude = np.abs(np.fft.rfft(batch, axis=1)) * scale
        mixed = magnitude[:, lower] * (1 - weight) + magnitude[:, lower + 1] * weight
        for row in np.flatnonzero(stop > first):
            covered = magnitude[:, first[row] : stop[row]].max(axis=1)
            mixed[:, row] = np.maximum(mixed[:, row], covered)
        decibels = 20 * np.log10(mixed + 1e-12)
        out[start : start + _BATCH] = np.clip(1 + decibels / range_db, 0.0, 1.0)
    return out


def peak_envelope(samples: npt.ArrayLike, width: int) -> tuple[Samples, Samples]:
    """The lowest and highest sample in each of `width` equal slices.

    Drawn as vertical lines, one per pixel column, they outline the whole
    signal. A signal shorter than `width` repeats its samples."""
    x = np.asarray(samples, dtype=float)
    if width <= 0:
        raise ValueError("width must be positive")
    if len(x) == 0:
        return np.zeros(width), np.zeros(width)
    edges = np.linspace(0, len(x), width + 1).astype(int)
    starts = np.minimum(edges[:-1], len(x) - 1)
    stops = np.maximum(edges[1:], starts + 1)
    low = np.array([x[a:b].min() for a, b in zip(starts, stops, strict=True)])
    high = np.array([x[a:b].max() for a, b in zip(starts, stops, strict=True)])
    return low, high


class _Stream(Protocol):
    """What `LiveSpectrogram` reads: a `daw.live.RingBuffer` is one"""

    capacity: int

    @property
    def written(self) -> int: ...

    def latest(self, count: int) -> Samples: ...


class LiveSpectrogram:
    """A spectrogram that grows as samples arrive, for a display of live sound.

    `update(stream)` looks at what has been written to the stream (a
    `RingBuffer`) since the last time and adds a column for every `HOP`
    samples. The columns are the same as `spectrogram_columns` would give for
    everything written so far, so a live display and a finished one agree.
    `levels` holds the newest `max_columns` of them, oldest first. If more
    arrives between two updates than the stream can hold, the oldest part is
    skipped."""

    def __init__(
        self,
        sample_rate: int,
        rows: int = 256,
        max_columns: int = 1024,
        *,
        window: int = WINDOW,
        hop: int = HOP,
    ) -> None:
        if max_columns < 1:
            raise ValueError("max_columns must be positive")
        if window < 2 or hop < 1 or window % hop:
            raise ValueError("window must be a whole number of hops")
        self.sample_rate = sample_rate
        self.rows = rows
        self.max_columns = max_columns
        self.window = window
        self.hop = hop
        self._levels = np.zeros((0, rows))
        self._hops_seen = 0

    @property
    def levels(self) -> Samples:
        return self._levels

    def update(self, stream: _Stream) -> int:
        """Add the columns for the new samples; returns how many were added"""
        hop, window = self.hop, self.window
        hops_now = stream.written // hop
        fresh = hops_now - self._hops_seen
        if fresh <= 0:
            return 0
        self._hops_seen = hops_now
        spare = stream.written % hop  # samples after the last whole hop
        room = (stream.capacity - spare - (window - hop)) // hop
        if room < 1:
            raise ValueError("the stream is too small to hold one window")
        fresh = min(fresh, room)
        span = (window - hop) + fresh * hop
        # The newest column's window must end on a whole hop, not on `written`
        samples = stream.latest(span + spare)[:span]
        columns = spectrogram_columns(
            samples, self.sample_rate, self.rows, window=window, hop=hop
        )[-fresh:]
        self._levels = np.concatenate([self._levels, columns])[-self.max_columns :]
        return fresh
