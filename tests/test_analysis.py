# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""The numbers behind the spectrogram and the signal overview"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from symsound.analysis import (
    HOP,
    WINDOW,
    LiveSpectrogram,
    frequency_of_row,
    peak_envelope,
    row_of_frequency,
    spectrogram_columns,
)

RATE = 8000


def tone(frequency, seconds=1.0, amplitude=1.0, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    return amplitude * np.sin(2 * np.pi * frequency * t)


class TestRows:
    def test_the_bottom_row_is_the_lowest_frequency_and_the_top_is_near_nyquist(self):
        assert frequency_of_row(0, 100, RATE) == pytest.approx(30.0)
        assert frequency_of_row(99, 100, RATE) == pytest.approx(RATE / 2)

    def test_every_octave_has_the_same_height(self):
        rows = 301
        a = row_of_frequency(200, rows, RATE) - row_of_frequency(100, rows, RATE)
        b = row_of_frequency(2000, rows, RATE) - row_of_frequency(1000, rows, RATE)
        assert a == pytest.approx(b)

    @given(st.floats(min_value=0, max_value=63))
    def test_the_two_directions_undo_each_other(self, row):
        frequency = frequency_of_row(row, 64, 44100)
        assert row_of_frequency(frequency, 64, 44100) == pytest.approx(row, abs=1e-9)

    def test_a_custom_lowest_frequency(self):
        assert frequency_of_row(0, 10, RATE, low=100.0) == pytest.approx(100.0)
        assert row_of_frequency(100.0, 10, RATE, low=100.0) == pytest.approx(0.0)

    def test_one_row_is_not_a_spectrogram(self):
        with pytest.raises(ValueError, match="two rows"):
            frequency_of_row(0, 1, RATE)
        with pytest.raises(ValueError, match="two rows"):
            row_of_frequency(100, 1, RATE)


class TestSpectrogramColumns:
    @pytest.mark.parametrize("frequency", [110, 440, 1000, 3000])
    def test_a_pure_tone_is_brightest_at_its_row(self, frequency):
        rows = 256
        grid = spectrogram_columns(tone(frequency), RATE, rows)
        expected = row_of_frequency(frequency, rows, RATE)
        middle = grid[len(grid) // 2]
        assert np.argmax(middle) == pytest.approx(expected, abs=2.5)

    def test_a_full_scale_sine_reaches_the_top_of_the_scale(self):
        grid = spectrogram_columns(tone(1000), RATE, 256)
        assert grid[len(grid) // 2].max() == pytest.approx(1.0, abs=0.05)

    @given(
        st.floats(min_value=60, max_value=3900),
        st.sampled_from([48, 100, 256]),
    )
    def test_a_tone_never_falls_between_two_rows(self, frequency, rows):
        # Regression: rows wider than a frequency bin used to skip the peak
        # of a narrow tone, which then looked up to 8 dB too quiet
        grid = spectrogram_columns(tone(frequency, 0.6), RATE, rows)
        assert grid[len(grid) // 2].max() > 0.9

    def test_the_scale_is_in_decibels_below_full_scale(self):
        full = spectrogram_columns(tone(1000), RATE, 256, range_db=80)
        half = spectrogram_columns(tone(1000, amplitude=0.1), RATE, 256, range_db=80)
        middle = len(full) // 2
        # A tenth of the amplitude is 20 dB down, a quarter of an 80 dB range
        drop = full[middle].max() - half[middle].max()
        assert drop == pytest.approx(0.25, abs=0.05)

    def test_a_quiet_signal_looks_quiet_not_normalised_to_full_height(self):
        quiet = spectrogram_columns(tone(1000, amplitude=0.001), RATE, 256)
        assert quiet.max() < 0.5

    def test_silence_is_the_bottom_of_the_scale(self):
        grid = spectrogram_columns(np.zeros(5000), RATE, 64)
        assert (grid == 0).all()

    def test_values_stay_between_zero_and_one(self):
        loud = spectrogram_columns(10 * tone(500), RATE, 64)
        assert loud.min() >= 0 and loud.max() <= 1

    @given(st.integers(min_value=1, max_value=6000))
    def test_there_is_one_column_per_hop_of_signal(self, length):
        grid = spectrogram_columns(np.zeros(length), RATE, 8)
        assert grid.shape == (-(-length // HOP), 8)

    def test_an_empty_signal_has_no_columns(self):
        assert spectrogram_columns([], RATE, 16).shape == (0, 16)

    def test_a_column_ends_at_its_hop_so_the_playhead_column_is_position_over_hop(
        self,
    ):
        # Silence, then a loud tone from sample 4096: columns before it are
        # silent, because their windows end before the tone starts
        signal = np.concatenate([np.zeros(4096), tone(1000, 0.5)])
        grid = spectrogram_columns(signal, RATE, 128)
        first_loud = int(np.argmax(grid.max(axis=1) > 0.2))
        assert first_loud == pytest.approx(4096 // HOP, abs=1)
        assert grid[: 4096 // HOP - 1].max() == 0

    def test_a_chirp_climbs_the_rows(self):
        t = np.arange(RATE * 2) / RATE
        signal = np.sin(2 * np.pi * (200 * t + 400 * t**2))  # 200 -> 1800 Hz
        grid = spectrogram_columns(signal, RATE, 200)
        peaks = grid.argmax(axis=1)[4:-4]
        assert (np.diff(peaks[::8]) >= 0).all()
        assert peaks[-1] > peaks[0] + 30

    def test_two_tones_make_two_bright_rows(self):
        grid = spectrogram_columns(tone(300) + tone(2000), RATE, 256)
        column = grid[len(grid) // 2]
        found = sorted(np.argsort(column)[-2:])
        low, high = row_of_frequency(300, 256, RATE), row_of_frequency(2000, 256, RATE)
        assert found[0] == pytest.approx(low, abs=3)
        assert found[1] == pytest.approx(high, abs=3)

    def test_a_long_signal_gives_the_same_columns_as_the_pieces_it_is_cut_into(self):
        # Columns are made in batches; the batching must not show
        signal = tone(700, 40.0)
        whole = spectrogram_columns(signal, RATE, 32)
        assert len(whole) > 512
        np.testing.assert_allclose(
            whole[:100], spectrogram_columns(signal[: 100 * HOP], RATE, 32)
        )

    def test_window_and_hop_can_change(self):
        grid = spectrogram_columns(tone(500), RATE, 16, window=512, hop=256)
        assert grid.shape == (RATE // 256 + (RATE % 256 > 0), 16)

    def test_the_defaults(self):
        assert (WINDOW, HOP) == (2048, 512)

    @pytest.mark.parametrize(
        "options",
        [
            {"window": 1},
            {"hop": 0},
            {"rows": 1},
            {"range_db": 0},
        ],
    )
    def test_nonsense_settings_are_refused(self, options):
        with pytest.raises(ValueError, match="positive"):
            spectrogram_columns(tone(100, 0.1), RATE, **options)

    def test_a_bad_sample_rate_is_refused(self):
        with pytest.raises(ValueError, match="sample rate"):
            spectrogram_columns(tone(100, 0.1), 0, 16)

    def test_two_dimensional_input_is_refused(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            spectrogram_columns(np.zeros((4, 4)), RATE, 8)


class TestPeakEnvelope:
    def test_each_slice_reports_its_lowest_and_highest_sample(self):
        low, high = peak_envelope([1, 5, 2, -3, 0, 4], 3)
        assert low.tolist() == [1, -3, 0]
        assert high.tolist() == [5, 2, 4]

    def test_it_outlines_the_signal(self):
        signal = tone(50, 2.0) * np.linspace(0, 1, 2 * RATE)
        low, high = peak_envelope(signal, 40)
        assert (high[1:] >= high[:-1] - 0.05).all()  # grows with the fade-in
        assert high[-1] == pytest.approx(1.0, abs=0.05)
        assert low[-1] == pytest.approx(-1.0, abs=0.05)

    @given(
        st.lists(st.floats(-1, 1), min_size=1, max_size=300),
        st.integers(min_value=1, max_value=80),
    )
    def test_it_always_brackets_the_signal(self, samples, width):
        low, high = peak_envelope(samples, width)
        assert low.shape == high.shape == (width,)
        assert (low <= high).all()
        assert low.min() == min(samples)
        assert high.max() == max(samples)

    def test_a_signal_shorter_than_the_width_repeats_its_samples(self):
        low, high = peak_envelope([2.0, -1.0], 6)
        assert set(low.tolist()) <= {2.0, -1.0}
        assert low.shape == (6,)

    def test_an_empty_signal_is_flat(self):
        low, high = peak_envelope([], 5)
        assert (low == 0).all() and (high == 0).all()

    def test_a_non_positive_width_is_refused(self):
        with pytest.raises(ValueError, match="width"):
            peak_envelope([1, 2, 3], 0)


class TestLiveSpectrogram:
    def stream(self, capacity=16384):
        from symsound.live import RingBuffer

        return RingBuffer(capacity)

    def test_it_starts_empty(self):
        live = LiveSpectrogram(RATE, rows=32)
        assert live.levels.shape == (0, 32)

    def test_nothing_new_adds_nothing(self):
        live, stream = LiveSpectrogram(RATE, rows=16), self.stream()
        assert live.update(stream) == 0
        stream.write(np.zeros(HOP - 1))
        assert live.update(stream) == 0
        assert live.levels.shape[0] == 0

    def test_a_column_arrives_for_every_hop_written(self):
        live, stream = LiveSpectrogram(RATE, rows=16), self.stream()
        stream.write(tone(500, 0.2))  # 1600 samples = 3 hops and a bit
        assert live.update(stream) == 3
        stream.write(np.zeros(HOP))
        assert live.update(stream) == 1
        assert live.levels.shape == (4, 16)

    @given(
        st.lists(st.integers(min_value=0, max_value=1500), min_size=1, max_size=10),
        st.integers(min_value=0, max_value=1000),
    )
    def test_it_agrees_with_the_finished_spectrogram_of_the_same_sound(
        self, sizes, seed
    ):
        # However the samples arrive, the columns are those of the whole signal
        rng = np.random.default_rng(seed)
        signal = rng.uniform(-1, 1, sum(sizes))
        live, stream = LiveSpectrogram(RATE, rows=24, max_columns=10_000), self.stream()
        position = 0
        for size in sizes:
            stream.write(signal[position : position + size])
            position += size
            live.update(stream)
        whole = spectrogram_columns(signal, RATE, 24)
        finished = len(signal) // HOP
        np.testing.assert_allclose(live.levels, whole[:finished], atol=1e-9)

    def test_a_tone_lights_its_row_as_it_arrives(self):
        live, stream = LiveSpectrogram(RATE, rows=128), self.stream()
        stream.write(tone(1000, 1.0))
        live.update(stream)
        expected = row_of_frequency(1000, 128, RATE)
        assert np.argmax(live.levels[-1]) == pytest.approx(expected, abs=2.5)

    def test_only_the_newest_columns_are_kept(self):
        live, stream = LiveSpectrogram(RATE, rows=8, max_columns=5), self.stream()
        stream.write(np.arange(HOP * 12) % 7 / 7.0)
        live.update(stream)
        whole = spectrogram_columns(np.arange(HOP * 12) % 7 / 7.0, RATE, 8)
        assert live.levels.shape == (5, 8)
        np.testing.assert_allclose(live.levels, whole[7:12], atol=1e-9)

    def test_if_more_arrives_than_the_stream_holds_the_oldest_part_is_skipped(self):
        capacity = 4096
        live, stream = LiveSpectrogram(RATE, rows=8), self.stream(capacity)
        signal = np.random.default_rng(3).uniform(-1, 1, HOP * 40)
        stream.write(signal)
        added = live.update(stream)
        assert 0 < added < 40
        whole = spectrogram_columns(signal, RATE, 8)
        # What it did add is the end of the finished spectrogram
        np.testing.assert_allclose(live.levels, whole[-added:], atol=1e-9)
        # ...and it carries on correctly from there
        more = np.random.default_rng(4).uniform(-1, 1, HOP * 2)
        stream.write(more)
        assert live.update(stream) == 2
        everything = spectrogram_columns(np.concatenate([signal, more]), RATE, 8)
        np.testing.assert_allclose(live.levels[-2:], everything[-2:], atol=1e-9)

    def test_a_stream_too_small_for_one_window_is_refused(self):
        live, stream = LiveSpectrogram(RATE, rows=8), self.stream(1024)
        stream.write(np.zeros(HOP * 3))
        with pytest.raises(ValueError, match="too small"):
            live.update(stream)

    def test_window_and_hop_can_change(self):
        live = LiveSpectrogram(RATE, rows=8, window=512, hop=256)
        stream = self.stream(4096)
        signal = np.random.default_rng(1).uniform(-1, 1, 256 * 9)
        stream.write(signal)
        live.update(stream)
        whole = spectrogram_columns(signal, RATE, 8, window=512, hop=256)
        np.testing.assert_allclose(live.levels, whole, atol=1e-9)

    @pytest.mark.parametrize(
        "options",
        [{"max_columns": 0}, {"window": 1000, "hop": 512}, {"window": 1}, {"hop": 0}],
    )
    def test_nonsense_settings_are_refused(self, options):
        with pytest.raises(ValueError):
            LiveSpectrogram(RATE, **options)
