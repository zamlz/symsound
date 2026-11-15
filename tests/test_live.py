# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""The live engine: notes rendered in pieces, mixed, and sent to the speakers

The reference for every rendering test is `compile`, which builds the whole
signal at once. Rendering in pieces must give the same samples.
"""

import os
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

import daw
from daw import live
from daw.constants import DEFAULT_SAMPLE_RATE, Note
from daw.envelope import ACTIVE_DURATION, adsr_envelope
from daw.filters import low_pass_filter
from daw.live import (
    Instrument,
    LiveOutput,
    Mixer,
    NoteSource,
    Renderer,
    RingBuffer,
)
from daw.waves.basic import sawtooth, sine
from daw.waves.time import TIME
from daw.waves.utils import WHITE_NOISE, compile
from helpers import evaluate

RATE = 8000
ENVELOPE = adsr_envelope(0.02, 0.05, 0.6, 0.1)
CHORD_TONE = sine(440) * ENVELOPE


def render_in_chunks(mixer, sizes):
    return np.concatenate([mixer.render(size) for size in sizes])


def frequency_of(samples, rate=RATE):
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return np.argmax(spectrum) * rate / len(samples)


class TestNoteSource:
    @pytest.mark.parametrize("expression", [sine(440), sawtooth(220), CHORD_TONE])
    def test_a_whole_render_matches_compile(self, expression):
        expression = expression.subs(ACTIVE_DURATION, 1e9)
        source = NoteSource(expression, RATE)
        np.testing.assert_allclose(
            source.render(0, RATE // 2), compile(expression, 0.5, RATE), atol=1e-9
        )

    @given(
        st.lists(st.integers(min_value=0, max_value=700), min_size=1, max_size=8),
    )
    def test_pieces_join_up_to_the_whole_signal(self, sizes):
        expression = (sine(330) + 0.5 * sawtooth(110)) * TIME
        source = NoteSource(expression, RATE)
        pieces, position = [], 0
        for size in sizes:
            pieces.append(source.render(position, size))
            position += size
        whole = evaluate(expression, np.arange(position) / RATE)
        np.testing.assert_allclose(np.concatenate(pieces), whole, atol=1e-9)

    def test_rendering_starts_wherever_asked(self):
        source = NoteSource(sine(500), RATE)
        np.testing.assert_allclose(
            source.render(1234, 100),
            compile(sine(500), 0.3, RATE)[1234:1334],
            atol=1e-9,
        )

    def test_the_envelope_follows_the_active_duration(self):
        source = NoteSource(CHORD_TONE, RATE)
        for held in (0.02, 0.1, 0.5):
            expected = compile(CHORD_TONE.subs(ACTIVE_DURATION, held), 0.8, RATE)
            np.testing.assert_allclose(
                source.render(0, 6400, held), expected, atol=1e-9
            )

    def test_an_unreleased_note_stays_in_its_sustain(self):
        samples = NoteSource(ENVELOPE, RATE).render(0, 4 * RATE)
        assert samples[-1] == pytest.approx(0.6)
        assert (samples[1:] > 0).all()

    def test_noise_is_fresh_in_every_piece_and_spans_the_range(self):
        source = NoteSource(WHITE_NOISE, RATE)
        first, second = source.render(0, 20000), source.render(20000, 20000)
        assert not np.array_equal(first, second)
        assert first.min() >= -1 and first.max() <= 1
        assert first.min() < -0.99 and first.max() > 0.99
        assert abs(first.mean()) < 0.03

    def test_noise_follows_numpys_seed(self):
        source = NoteSource(WHITE_NOISE, RATE)
        np.random.seed(5)
        one = source.render(0, 50)
        np.random.seed(5)
        two = source.render(0, 50)
        np.testing.assert_array_equal(one, two)

    @pytest.mark.parametrize("value", [0, 0.25, -1])
    def test_a_constant_gives_one_value_per_sample(self, value):
        samples = NoteSource(sp.Float(value), RATE).render(0, 37)
        assert samples.shape == (37,)
        assert (samples == value).all()

    def test_an_empty_piece_is_empty(self):
        assert NoteSource(sine(440), RATE).render(10, 0).shape == (0,)

    def test_the_result_can_be_changed(self):
        samples = NoteSource(sp.Float(0.5), RATE).render(0, 4)
        samples[0] = 3.0
        assert samples[0] == 3.0

    def test_a_filtered_signal_is_refused_with_a_reason(self):
        filtered = low_pass_filter(sine(440), 1000)
        with pytest.raises(ValueError, match="filtered signal cannot be rendered"):
            NoteSource(filtered, RATE)

    def test_a_filter_hidden_inside_a_larger_expression_is_refused(self):
        filtered = 0.5 * sine(110) + 0.5 * low_pass_filter(sine(440), 1000)
        with pytest.raises(ValueError, match="filtered"):
            NoteSource(filtered, RATE)

    @pytest.mark.parametrize("rate", [0, -44100])
    def test_a_bad_sample_rate_is_refused(self, rate):
        with pytest.raises(ValueError, match="sample rate"):
            NoteSource(sine(440), rate)

    @pytest.mark.parametrize("first, count", [(-1, 10), (0, -1)])
    def test_negative_positions_are_refused(self, first, count):
        with pytest.raises(ValueError, match="negative"):
            NoteSource(sine(440), RATE).render(first, count)

    def test_it_knows_whether_the_signal_has_a_release(self):
        assert NoteSource(CHORD_TONE, RATE).has_release
        assert not NoteSource(sine(440), RATE).has_release

    def test_a_python_number_is_accepted_as_an_expression(self):
        assert NoteSource(0.25, RATE).render(0, 3).tolist() == [0.25] * 3  # type: ignore[arg-type]

    def test_the_default_sample_rate_is_the_projects(self):
        assert NoteSource(sine(440)).sample_rate == DEFAULT_SAMPLE_RATE


class TestMixerSounds:
    def test_a_held_note_matches_compile(self):
        mixer = Mixer(gain=1.0)
        mixer.note_on("a", NoteSource(CHORD_TONE, RATE))
        heard = render_in_chunks(mixer, [512] * 10)
        expected = compile(CHORD_TONE.subs(ACTIVE_DURATION, 1e9), 5120 / RATE, RATE)
        np.testing.assert_allclose(heard, expected, atol=1e-9)

    @given(st.lists(st.integers(min_value=1, max_value=900), min_size=1, max_size=6))
    def test_the_chunk_size_does_not_change_the_sound(self, sizes):
        whole = Mixer(gain=1.0)
        whole.note_on("a", NoteSource(sine(300), RATE))
        pieces = Mixer(gain=1.0)
        pieces.note_on("a", NoteSource(sine(300), RATE))
        np.testing.assert_allclose(
            whole.render(sum(sizes)), render_in_chunks(pieces, sizes), atol=1e-9
        )

    def test_a_note_is_heard_at_its_pitch(self):
        mixer = Mixer()
        mixer.note_on(Note.A4, NoteSource(sine(Note.A4), RATE))
        assert frequency_of(mixer.render(RATE)) == pytest.approx(440, abs=2)

    def test_a_chord_contains_each_of_its_notes(self):
        mixer = Mixer()
        for note in (Note.C4, Note.E4, Note.G4):
            mixer.note_on(note, NoteSource(sine(note), RATE))
        spectrum = np.abs(np.fft.rfft(mixer.render(RATE) * np.hanning(RATE)))
        strongest = sorted(np.argsort(spectrum)[-3:])
        for found, note in zip(strongest, (Note.C4, Note.E4, Note.G4), strict=True):
            assert found == pytest.approx(note.value, abs=2)

    def test_notes_add_up(self):
        low, high = sine(200), sine(300)
        mixer = Mixer(gain=1.0)
        mixer.note_on("low", NoteSource(low, RATE))
        mixer.note_on("high", NoteSource(high, RATE))
        np.testing.assert_allclose(
            mixer.render(2000), compile(low + high, 0.25, RATE), atol=1e-9
        )

    def test_gain_scales_the_mix(self):
        loud, quiet = Mixer(gain=1.0), Mixer(gain=0.25)
        for mixer in (loud, quiet):
            mixer.note_on("a", NoteSource(sine(200), RATE))
        np.testing.assert_allclose(quiet.render(500), 0.25 * loud.render(500))

    def test_the_default_gain_leaves_room_for_two_notes(self):
        mixer = Mixer()
        for note in (Note.C4, Note.G4):
            mixer.note_on(note, NoteSource(sine(note), RATE))
        assert np.abs(mixer.render(RATE)).max() <= 1.0

    def test_a_bigger_chord_needs_a_smaller_gain(self):
        notes = (Note.C4, Note.E4, Note.G4, Note.C5)
        peaks = []
        for gain in (0.5, 0.25):
            mixer = Mixer(gain)
            for note in notes:
                mixer.note_on(note, NoteSource(sine(note), RATE))
            peaks.append(np.abs(mixer.render(RATE)).max())
        assert peaks[0] > 1.0
        assert peaks[1] <= 1.0

    def test_nothing_playing_is_silence(self):
        assert (Mixer().render(300) == 0).all()

    def test_a_note_that_starts_later_begins_at_its_own_time_zero(self):
        mixer = Mixer(gain=1.0)
        mixer.render(777)
        mixer.note_on("a", NoteSource(TIME + 1, RATE))
        heard = mixer.render(5)
        np.testing.assert_allclose(heard, 1 + np.arange(5) / RATE)

    def test_a_note_in_its_first_chunk_is_the_start_of_the_signal(self):
        mixer = Mixer(gain=1.0)
        mixer.note_on("a", NoteSource(sine(440) + 2, RATE))
        assert mixer.render(1)[0] == pytest.approx(2.0)

    def test_a_note_can_be_played_twice_and_sounds_the_same_each_time(self):
        mixer = Mixer(gain=1.0)
        source = NoteSource(sine(440) * ENVELOPE, RATE)
        takes = []
        for _ in range(2):
            mixer.note_on("a", source)
            held = mixer.render(1600)
            mixer.note_off("a")
            mixer.render(RATE)
            takes.append(held)
        np.testing.assert_allclose(takes[0], takes[1], atol=1e-9)

    def test_empty_and_negative_chunks(self):
        mixer = Mixer()
        assert mixer.render(0).shape == (0,)
        with pytest.raises(ValueError, match="negative"):
            mixer.render(-1)

    def test_a_negative_gain_is_refused(self):
        with pytest.raises(ValueError, match="gain"):
            Mixer(gain=-0.1)

    def test_zero_gain_is_silent(self):
        mixer = Mixer(gain=0)
        mixer.note_on("a", NoteSource(sine(440), RATE))
        assert (mixer.render(100) == 0).all()


class TestMixerKeys:
    def test_a_key_is_held_from_press_to_release(self):
        mixer = Mixer()
        assert not mixer.is_held("a")
        mixer.note_on("a", NoteSource(CHORD_TONE, RATE))
        assert mixer.is_held("a")
        mixer.note_off("a")
        assert not mixer.is_held("a")

    def test_pressing_a_held_key_again_does_nothing(self):
        mixer = Mixer(gain=1.0)
        source = NoteSource(sine(440), RATE)
        mixer.note_on("a", source)
        mixer.render(100)
        mixer.note_on("a", source)  # key auto-repeat
        assert mixer.active == 1
        np.testing.assert_allclose(
            mixer.render(100), compile(sine(440), 0.05, RATE)[100:200], atol=1e-9
        )

    def test_releasing_a_key_that_was_never_pressed_does_nothing(self):
        mixer = Mixer()
        mixer.note_off("nothing")
        assert mixer.active == 0

    def test_releasing_twice_is_the_same_as_once(self):
        once, twice = Mixer(gain=1.0), Mixer(gain=1.0)
        for mixer in (once, twice):
            mixer.note_on("a", NoteSource(CHORD_TONE, RATE))
            mixer.render(500)
        once.note_off("a")
        twice.note_off("a")
        twice.render(0)
        twice.note_off("a")
        np.testing.assert_allclose(once.render(1500), twice.render(1500))

    def test_keys_are_independent(self):
        mixer = Mixer(gain=1.0)
        mixer.note_on("a", NoteSource(sine(200), RATE))
        mixer.note_on("b", NoteSource(sine(300), RATE))
        mixer.note_off("a")
        assert mixer.is_held("b") and not mixer.is_held("a")

    def test_pressing_a_key_that_is_dying_away_starts_a_new_note(self):
        mixer = Mixer(gain=1.0)
        source = NoteSource(CHORD_TONE, RATE)
        mixer.note_on("a", source)
        mixer.render(800)
        mixer.note_off("a")
        mixer.render(100)  # still in its release
        mixer.note_on("a", source)
        assert mixer.active == 2
        assert mixer.is_held("a")
        mixer.note_off("a")
        assert not mixer.is_held("a")


class TestRelease:
    def test_a_released_note_follows_its_envelope_release(self):
        mixer = Mixer(gain=1.0)
        mixer.note_on("a", NoteSource(CHORD_TONE, RATE))
        held = 1600  # 0.2 s, on a chunk boundary
        first = mixer.render(held)
        mixer.note_off("a")
        tail = mixer.render(RATE // 2)
        expected = compile(CHORD_TONE.subs(ACTIVE_DURATION, held / RATE), 0.7, RATE)
        np.testing.assert_allclose(np.concatenate([first, tail]), expected, atol=1e-9)

    def test_the_release_is_audible_then_gone_when_the_envelope_ends(self):
        mixer = Mixer(gain=1.0)
        mixer.note_on("a", NoteSource(CHORD_TONE, RATE))
        mixer.render(2400)
        mixer.note_off("a")
        tail = np.concatenate([mixer.render(400) for _ in range(5)])
        # The envelope's release is 0.1 s = 800 samples
        assert np.abs(tail[:700]).max() > 0.05
        assert (tail[820:] == 0).all()

    def test_the_note_is_dropped_once_it_falls_silent(self):
        mixer = Mixer()
        mixer.note_on("a", NoteSource(CHORD_TONE, RATE))
        mixer.render(2400)
        mixer.note_off("a")
        counts = []
        for _ in range(5):
            mixer.render(400)
            counts.append(mixer.active)
        assert counts[0] == 1
        assert counts[-1] == 0

    def test_a_note_with_no_envelope_fades_out_without_a_click(self):
        mixer = Mixer(gain=1.0)
        mixer.note_on("a", NoteSource(sine(440) + 1, RATE))
        mixer.render(1000)
        mixer.note_off("a")
        tail = mixer.render(500)
        fade = int(live.FADE_OUT * RATE)
        ramp = 1 - np.arange(fade) / fade
        expected = (
            1 + np.sin(2 * np.pi * 440 * (1000 + np.arange(fade)) / RATE)
        ) * ramp
        np.testing.assert_allclose(tail[:fade], expected, atol=1e-9)
        assert (tail[fade:] == 0).all()
        assert mixer.active == 0

    def test_a_fade_is_gradual_and_never_louder_than_the_note(self):
        mixer = Mixer(gain=1.0)
        mixer.note_on("a", NoteSource(sp.Float(1.0), RATE))
        mixer.render(100)
        mixer.note_off("a")
        tail = mixer.render(200)
        fade = int(live.FADE_OUT * RATE)
        assert tail[0] == 1.0
        assert (np.diff(tail[: fade + 1]) < 0).all()
        assert tail.max() <= 1.0 and tail.min() >= 0.0

    def test_a_fade_spread_over_several_small_chunks_is_the_same_fade(self):
        small, big = Mixer(gain=1.0), Mixer(gain=1.0)
        for mixer in (small, big):
            mixer.note_on("a", NoteSource(sp.Float(1.0), RATE))
            mixer.render(64)
            mixer.note_off("a")
        np.testing.assert_allclose(render_in_chunks(small, [7] * 20), big.render(140))

    def test_an_empty_chunk_does_not_end_a_release(self):
        # Regression: zero samples are "all zeros", which looked like silence
        mixer = Mixer(gain=1.0)
        mixer.note_on("a", NoteSource(CHORD_TONE, RATE))
        mixer.render(2400)
        mixer.note_off("a")
        mixer.render(0)
        assert mixer.active == 1
        assert np.abs(mixer.render(200)).max() > 0.05

    def test_releasing_before_the_first_chunk_still_ends_cleanly(self):
        mixer = Mixer(gain=1.0)
        mixer.note_on("a", NoteSource(CHORD_TONE, RATE))
        mixer.note_off("a")
        mixer.render(RATE)
        assert mixer.active == 0

    def test_a_release_that_never_goes_silent_is_cut_off(self, monkeypatch):
        monkeypatch.setattr(live, "LONGEST_RELEASE", 0.05)
        never_silent = sine(440) * sp.Min(1, ACTIVE_DURATION + 1)
        mixer = Mixer()
        mixer.note_on("a", NoteSource(never_silent, RATE))
        mixer.render(100)
        mixer.note_off("a")
        mixer.render(200)
        assert mixer.active == 1
        mixer.render(400)
        assert mixer.active == 0

    @given(
        st.integers(min_value=1, max_value=20),
        st.lists(st.integers(min_value=64, max_value=600), min_size=1, max_size=5),
    )
    def test_releasing_after_any_number_of_chunks_matches_compile(self, held, sizes):
        mixer = Mixer(gain=1.0)
        mixer.note_on("a", NoteSource(CHORD_TONE, RATE))
        held_samples = held * 100
        first = mixer.render(held_samples)
        mixer.note_off("a")
        rest = render_in_chunks(mixer, sizes)
        total = held_samples + len(rest)
        # Evaluated at i / RATE directly: compile's grid can be a sample short
        expected = evaluate(
            CHORD_TONE.subs(ACTIVE_DURATION, held_samples / RATE),
            np.arange(total) / RATE,
        )
        np.testing.assert_allclose(np.concatenate([first, rest]), expected, atol=1e-9)


class TestInstrument:
    def test_a_note_is_compiled_once_and_remembered(self):
        calls = []

        def function(note):
            calls.append(note)
            return sine(note)

        instrument = Instrument(function, RATE)
        first = instrument.source(Note.A4)
        assert instrument.source(Note.A4) is first
        assert calls == [Note.A4]

    def test_each_note_sounds_at_its_own_pitch(self):
        instrument = Instrument(lambda note: sine(note), RATE)
        for note in (Note.A3, Note.A4, Note.E5):
            heard = instrument.source(note).render(0, RATE)
            assert frequency_of(heard) == pytest.approx(note.value, abs=2)

    def test_precompile_makes_every_source_before_any_key_is_pressed(self):
        calls = []
        instrument = Instrument(lambda note: calls.append(note) or sine(note), RATE)
        instrument.precompile([Note.C4, Note.D4, Note.C4])
        assert calls == [Note.C4, Note.D4]
        instrument.source(Note.D4)
        assert calls == [Note.C4, Note.D4]

    def test_the_sources_use_the_instruments_sample_rate(self):
        assert Instrument(sine, RATE).source(Note.A4).sample_rate == RATE

    def test_the_default_sample_rate_is_the_projects(self):
        assert Instrument(sine).sample_rate == DEFAULT_SAMPLE_RATE

    def test_a_filtered_instrument_fails_when_the_note_is_asked_for(self):
        instrument = Instrument(lambda note: low_pass_filter(sine(note), 800), RATE)
        with pytest.raises(ValueError, match="filtered"):
            instrument.source(Note.A4)

    def test_every_note_of_the_piano_compiles_fast_enough_to_do_at_start_up(self):
        import time

        instrument = Instrument(lambda note: sine(note) * ENVELOPE, RATE)
        started = time.perf_counter()
        instrument.precompile(list(Note))
        assert time.perf_counter() - started < 20

    def test_rendering_a_chunk_is_far_faster_than_playing_it(self):
        import time

        source = NoteSource(sine(440) * ENVELOPE, DEFAULT_SAMPLE_RATE)
        started = time.perf_counter()
        for index in range(50):
            source.render(index * 512, 512)
        per_chunk = (time.perf_counter() - started) / 50
        assert per_chunk < 0.5 * 512 / DEFAULT_SAMPLE_RATE


class TestRingBuffer:
    def test_it_starts_silent(self):
        ring = RingBuffer(8)
        assert ring.written == 0
        assert (ring.latest(8) == 0).all()

    def test_it_returns_what_was_written_oldest_first(self):
        ring = RingBuffer(8)
        ring.write([1, 2, 3])
        assert ring.latest(3).tolist() == [1, 2, 3]

    def test_fewer_samples_than_asked_for_are_padded_with_zeros_in_front(self):
        ring = RingBuffer(8)
        ring.write([1, 2, 3])
        assert ring.latest(5).tolist() == [0, 0, 1, 2, 3]

    def test_old_samples_are_overwritten(self):
        ring = RingBuffer(4)
        ring.write([1, 2, 3])
        ring.write([4, 5, 6])
        assert ring.latest(4).tolist() == [3, 4, 5, 6]
        assert ring.latest(2).tolist() == [5, 6]

    def test_a_write_longer_than_the_buffer_keeps_its_end(self):
        ring = RingBuffer(4)
        ring.write(np.arange(1, 11))
        assert ring.latest(4).tolist() == [7, 8, 9, 10]
        assert ring.written == 10

    def test_written_counts_every_sample(self):
        ring = RingBuffer(4)
        ring.write([1, 2, 3])
        ring.write([])
        ring.write([4, 5])
        assert ring.written == 5

    def test_latest_zero_is_empty(self):
        assert RingBuffer(4).latest(0).shape == (0,)

    @given(
        st.integers(min_value=1, max_value=40),
        st.lists(st.lists(st.floats(-1, 1), max_size=60), max_size=8),
    )
    def test_it_always_agrees_with_a_plain_list(self, capacity, writes):
        ring, reference = RingBuffer(capacity), deque(maxlen=capacity)
        for chunk in writes:
            ring.write(chunk)
            reference.extend(chunk)
        assert ring.written == sum(len(chunk) for chunk in writes)
        expected = np.zeros(capacity)
        expected[capacity - len(reference) :] = list(reference)
        np.testing.assert_array_equal(ring.latest(capacity), expected)
        half = capacity // 2
        np.testing.assert_array_equal(ring.latest(half), expected[capacity - half :])

    def test_a_copy_is_returned_not_the_inside(self):
        ring = RingBuffer(4)
        ring.write([1, 2, 3, 4])
        ring.latest(4)[:] = 99
        assert ring.latest(4).tolist() == [1, 2, 3, 4]

    @pytest.mark.parametrize("capacity", [0, -3])
    def test_a_bad_capacity_is_refused(self, capacity):
        with pytest.raises(ValueError, match="capacity"):
            RingBuffer(capacity)

    @pytest.mark.parametrize("count", [-1, 5])
    def test_asking_for_more_than_it_holds_is_refused(self, count):
        with pytest.raises(ValueError, match="count"):
            RingBuffer(4).latest(count)

    def test_a_writer_and_a_reader_in_different_threads_never_see_torn_data(self):
        ring = RingBuffer(256)
        stop = threading.Event()
        problems = []

        def write():
            value = 0
            while not stop.is_set():
                value += 1
                ring.write(np.full(64, float(value)))

        def read():
            for _ in range(300):
                seen = ring.latest(64)
                # Whole blocks of 64 are written, so the newest 64 are all
                # one value only when read between writes; they may span two
                # blocks, but values must never go backwards
                if (np.diff(seen) < 0).any():
                    problems.append(seen.copy())

        writer = threading.Thread(target=write)
        writer.start()
        read()
        stop.set()
        writer.join()
        assert not problems


class TestLiveOutput:
    def test_it_opens_a_mono_stream_with_the_given_settings(self, fake_sounddevice):
        LiveOutput(Mixer(), sample_rate=22050, block_size=256)
        (stream,) = fake_sounddevice.streams
        assert stream.options["samplerate"] == 22050
        assert stream.options["blocksize"] == 256
        assert stream.options["channels"] == 1
        assert stream.options["dtype"] == "float32"

    def test_the_defaults_are_the_projects_sample_rate_and_a_small_block(
        self, fake_sounddevice
    ):
        LiveOutput(Mixer())
        options = fake_sounddevice.streams[0].options
        assert options["samplerate"] == DEFAULT_SAMPLE_RATE
        assert options["blocksize"] == 512

    def test_the_speakers_receive_the_mix(self, fake_sounddevice):
        mixer = Mixer(gain=1.0)
        mixer.note_on("a", NoteSource(sine(440), RATE))
        LiveOutput(mixer, sample_rate=RATE)
        stream = fake_sounddevice.streams[0]
        outdata = np.zeros((512, 1), dtype=np.float32)
        stream.callback(outdata, 512, None, None)
        np.testing.assert_allclose(
            outdata[:, 0], compile(sine(440), 512 / RATE, RATE), atol=1e-6
        )

    def test_successive_blocks_continue_the_note(self, fake_sounddevice):
        mixer = Mixer(gain=1.0)
        mixer.note_on("a", NoteSource(sine(440), RATE))
        LiveOutput(mixer, sample_rate=RATE)
        stream = fake_sounddevice.streams[0]
        heard = []
        for _ in range(4):
            outdata = np.zeros((300, 1), dtype=np.float32)
            stream.callback(outdata, 300, None, None)
            heard.append(outdata[:, 0])
        np.testing.assert_allclose(
            np.concatenate(heard), compile(sine(440), 1200 / RATE, RATE), atol=1e-6
        )

    def test_silence_is_written_when_nothing_is_held(self, fake_sounddevice):
        LiveOutput(Mixer(), sample_rate=RATE)
        outdata = np.ones((128, 1), dtype=np.float32)
        fake_sounddevice.streams[0].callback(outdata, 128, None, None)
        assert (outdata == 0).all()

    def test_what_is_played_is_copied_to_the_tap(self, fake_sounddevice):
        mixer = Mixer(gain=1.0)
        mixer.note_on("a", NoteSource(sine(440), RATE))
        ring = RingBuffer(1000)
        LiveOutput(mixer, sample_rate=RATE, tap=ring)
        outdata = np.zeros((400, 1), dtype=np.float32)
        fake_sounddevice.streams[0].callback(outdata, 400, None, None)
        assert ring.written == 400
        np.testing.assert_allclose(ring.latest(400), outdata[:, 0], atol=1e-6)

    def test_start_and_stop_drive_the_stream(self, fake_sounddevice):
        output = LiveOutput(Mixer())
        output.start()
        output.stop()
        assert fake_sounddevice.streams[0].events == ["start", "stop", "close"]

    def test_as_a_context_manager_it_stops_even_after_an_error(self, fake_sounddevice):
        with pytest.raises(RuntimeError), LiveOutput(Mixer()):
            assert fake_sounddevice.streams[0].events == ["start"]
            raise RuntimeError("boom")
        assert fake_sounddevice.streams[0].events == ["start", "stop", "close"]

    def test_importing_the_module_does_not_need_an_audio_backend(self):
        code = (
            "import sys\n"
            "sys.modules['sounddevice'] = None\n"
            "import daw.live as live\n"
            "try:\n"
            "    live.LiveOutput(live.Mixer())\n"
            "except ImportError:\n"
            "    print('ok')\n"
        )
        src = str(Path(daw.__file__).resolve().parents[1])
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH")]))
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        assert result.stdout.strip() == "ok"


class TestThreads:
    def test_keys_pressed_while_the_audio_thread_renders_are_safe(self):
        mixer = Mixer()
        sources = [NoteSource(sine(100 + 50 * k) * ENVELOPE, RATE) for k in range(8)]
        stop = threading.Event()
        errors = []

        def audio():
            try:
                while not stop.is_set():
                    block = mixer.render(128)
                    assert np.isfinite(block).all()
            except Exception as error:  # noqa: BLE001 - reported below
                errors.append(error)

        thread = threading.Thread(target=audio)
        thread.start()
        for round_ in range(200):
            key = round_ % 8
            mixer.note_on(key, sources[key])
            mixer.note_off((round_ + 3) % 8)
        stop.set()
        thread.join()
        assert not errors


class TestPlayer:
    def player(self, length=1000, rate=RATE):
        return live.Player(np.arange(length, dtype=float), rate)

    def test_it_starts_paused_at_the_beginning(self):
        player = self.player()
        assert not player.playing
        assert player.position == 0
        assert player.seconds == 0.0
        assert not player.finished

    def test_it_knows_its_length(self):
        assert self.player(4000, 8000).duration == 0.5

    def test_paused_it_plays_silence_and_does_not_move(self):
        player = self.player()
        assert (player.render(100) == 0).all()
        assert player.position == 0

    def test_playing_hands_out_the_samples_in_order(self):
        player = self.player()
        player.play()
        np.testing.assert_array_equal(player.render(3), [0, 1, 2])
        np.testing.assert_array_equal(player.render(3), [3, 4, 5])
        assert player.position == 6

    @given(st.lists(st.integers(min_value=0, max_value=400), max_size=8))
    def test_any_chunking_plays_the_same_signal(self, sizes):
        player = self.player()
        player.play()
        heard = np.concatenate([player.render(size) for size in sizes] or [np.zeros(0)])
        expected = np.zeros(sum(sizes))
        expected[: min(len(expected), 1000)] = np.arange(min(len(expected), 1000))
        np.testing.assert_array_equal(heard, expected)

    def test_it_stops_by_itself_at_the_end_and_pads_with_silence(self):
        player = self.player(10)
        player.play()
        np.testing.assert_array_equal(player.render(14), list(range(10)) + [0] * 4)
        assert player.finished
        assert not player.playing
        assert (player.render(5) == 0).all()

    def test_pause_keeps_the_place_and_play_carries_on(self):
        player = self.player()
        player.play()
        player.render(10)
        player.pause()
        player.render(50)
        assert player.position == 10
        player.play()
        assert player.render(1)[0] == 10

    def test_toggle_flips_between_playing_and_paused(self):
        player = self.player()
        player.toggle()
        assert player.playing
        player.toggle()
        assert not player.playing

    def test_play_from_the_end_starts_again(self):
        player = self.player(10)
        player.play()
        player.render(10)
        assert player.finished
        player.play()
        assert player.position == 0
        assert player.render(2).tolist() == [0, 1]

    def test_seek_jumps_to_a_time_in_seconds(self):
        player = self.player(8000, 8000)
        player.seek(0.5)
        assert player.position == 4000
        assert player.seconds == 0.5
        player.play()
        assert player.render(1)[0] == 4000

    def test_seek_is_kept_between_the_ends(self):
        player = self.player(8000, 8000)
        player.seek(-3.0)
        assert player.position == 0
        player.seek(99.0)
        assert player.position == 8000
        assert player.finished

    def test_seeking_keeps_playing_state(self):
        player = self.player()
        player.play()
        player.seek(0.05)
        assert player.playing
        player.pause()
        player.seek(0.01)
        assert not player.playing

    def test_replay_goes_to_the_start_and_plays(self):
        player = self.player()
        player.play()
        player.render(500)
        player.replay()
        assert player.position == 0 and player.playing

    def test_the_signal_is_copied_so_later_changes_do_not_leak_in(self):
        data = np.ones(10)
        player = live.Player(data, RATE)
        data[:] = 5
        player.play()
        assert player.render(1)[0] == 1.0

    @pytest.mark.parametrize("samples", [[], [[1.0, 2.0]]])
    def test_it_needs_a_non_empty_one_dimensional_signal(self, samples):
        with pytest.raises(ValueError, match="non-empty"):
            live.Player(samples, RATE)

    def test_a_bad_sample_rate_is_refused(self):
        with pytest.raises(ValueError, match="sample rate"):
            live.Player([1.0], 0)

    def test_a_negative_chunk_is_refused(self):
        with pytest.raises(ValueError, match="negative"):
            self.player().render(-1)

    def test_a_mixer_and_a_player_both_count_as_renderers(self):
        # Renderer is the shape LiveOutput accepts: anything with render(count)
        def render_one(renderer: Renderer):
            return renderer.render(1)

        assert render_one(Mixer()).shape == (1,)
        assert render_one(live.Player([1.0, 2.0], RATE)).shape == (1,)

    def test_it_can_drive_the_speakers_like_a_mixer(self, fake_sounddevice):
        player = live.Player(compile(sine(440), 0.25, RATE), RATE)
        player.play()
        LiveOutput(player, sample_rate=RATE)
        outdata = np.zeros((400, 1), dtype=np.float32)
        fake_sounddevice.streams[0].callback(outdata, 400, None, None)
        np.testing.assert_allclose(
            outdata[:, 0], compile(sine(440), 0.25, RATE)[:400], atol=1e-6
        )
        assert player.position == 400

    def test_a_display_thread_can_read_the_position_while_audio_plays(self):
        player = self.player(200000)
        player.play()
        seen = []

        def audio():
            for _ in range(1000):
                player.render(100)

        thread = threading.Thread(target=audio)
        thread.start()
        while thread.is_alive():
            seen.append(player.position)
        thread.join()
        assert seen == sorted(seen)
        assert player.position == 100000
