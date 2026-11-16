# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""`delay` is a signal shifted later and silent before: a plain expression.

References are independent of symsound.effects: the signal's own reference
function evaluated at `t - seconds` (waves, envelopes), or the compiled
samples of the undelayed signal moved along by whole samples (anything with a
filter in it). The echo and reverb of docs/effects.md are built from `delay`
here too, and measured.
"""

import math

import numpy as np
import pytest
import sympy as sp
from hypothesis import assume, given
from hypothesis import strategies as st

from helpers import (
    away_from_edges,
    evaluate,
    ref_adsr,
    ref_sawtooth,
    ref_square,
    ref_triangle,
)
from symsound.effects import delay
from symsound.envelope import ACTIVE_DURATION, adsr_envelope
from symsound.filters import low_pass_filter
from symsound.waves.basic import sawtooth, sine, square, triangle
from symsound.waves.time import TIME
from symsound.waves.utils import WHITE_NOISE, compile

SR = 44100

frequencies = st.floats(min_value=20, max_value=2000)
seconds = st.floats(min_value=0.0, max_value=2.0)
times = st.lists(st.floats(min_value=0.0, max_value=4.0), min_size=1, max_size=30).map(
    np.array
)


def beep(frequency=500, length=0.05):
    """A sine that is on for `length` seconds from time zero, silent either side"""
    return sine(frequency) * sp.Piecewise((1, (TIME >= 0) & (TIME < length)), (0, True))


def shifted(samples, count):
    """The samples `count` places later, with silence in front"""
    return np.concatenate([np.zeros(count), samples[: len(samples) - count]])


def echo(signal, spacing, feedback, repeats):
    """The recipe of docs/effects.md"""
    return signal + sum(
        feedback**k * delay(signal, k * spacing) for k in range(1, repeats + 1)
    )


COMBS = (0.0253, 0.0269, 0.0289, 0.0307, 0.0322, 0.0338, 0.0353, 0.0367)


def reverb(signal, decay=1.0, level=0.3):
    """The recipe of docs/effects.md"""
    wet = 0
    for spacing in COMBS:
        gain = 10 ** (-3 * spacing / decay)
        wet += sum(
            gain**k * delay(signal, k * spacing)
            for k in range(1, int(decay / spacing) + 1)
        )
    return signal + level * wet / math.sqrt(len(COMBS))


class TestKnownValues:
    def test_a_sine_is_silent_until_the_delay_and_then_starts_from_zero(self):
        out = evaluate(delay(sine(1), 0.5), [0.0, 0.25, 0.4999, 0.5, 0.75, 1.0, 1.25])
        np.testing.assert_allclose(out, [0, 0, 0, 0, 1, 0, -1], atol=1e-3)

    def test_the_delay_is_exactly_zero_before_it_arrives(self):
        out = evaluate(delay(sine(440) + 1, 0.3), np.linspace(0, 0.2999, 500))
        assert not out.any()

    def test_a_signal_that_is_not_silent_before_zero_is_still_silent_before_the_delay(
        self,
    ):
        # sine(1) at negative time is -sin(...), not zero: the mask is what silences it
        assert evaluate(sine(1), [-0.25])[0] != 0
        assert evaluate(delay(sine(1), 0.5), [0.1])[0] == 0.0

    def test_a_delay_of_nothing_is_the_signal_itself(self):
        signal = sine(440)
        assert delay(signal, 0) is signal
        assert delay(signal, 0.0) is signal

    def test_a_click_on_the_grid_lands_a_delay_later(self):
        click = sp.Piecewise((1, (TIME >= 0) & (TIME < 1 / SR / 2)), (0, True))
        samples = compile(delay(click, 0.25), 0.5)
        assert np.flatnonzero(samples).tolist() == [11025]

    def test_the_delay_is_in_seconds_whatever_the_sample_rate(self):
        for rate in (22050, 44100, 48000):
            samples = compile(delay(beep(500, 0.01), 0.2), 0.5, rate)
            first = np.flatnonzero(np.abs(samples) > 0.01)[0] / rate
            assert first == pytest.approx(0.2, abs=2 / rate)

    def test_the_delay_may_be_an_exact_number(self):
        out = evaluate(delay(sine(1), sp.Rational(1, 2)), [0.75])
        assert out[0] == pytest.approx(1.0)


class TestAgainstReferences:
    @pytest.mark.parametrize(
        "wave, reference",
        [
            (square, ref_square),
            (triangle, ref_triangle),
            (sawtooth, ref_sawtooth),
        ],
    )
    @given(frequencies, seconds, times)
    def test_a_wave_is_the_wave_at_an_earlier_time(
        self, wave, reference, frequency, wait, t
    ):
        assume(np.all(np.abs(t - wait) > 1e-9))
        earlier = t - wait
        live = t > wait
        assume(away_from_edges(frequency, earlier[live]).all())
        out = evaluate(delay(wave(frequency), wait), t)
        expected = np.where(live, reference(frequency, np.maximum(earlier, 0)), 0.0)
        np.testing.assert_allclose(out, expected, atol=1e-9)

    @given(frequencies, seconds, times)
    def test_a_sine_is_the_sine_at_an_earlier_time(self, frequency, wait, t):
        assume(np.all(np.abs(t - wait) > 1e-9))
        out = evaluate(delay(sine(frequency), wait), t)
        expected = np.where(t > wait, np.sin(2 * np.pi * frequency * (t - wait)), 0.0)
        np.testing.assert_allclose(out, expected, atol=1e-9)

    @given(
        st.floats(0.001, 0.2),
        st.floats(0.0, 0.2),
        st.floats(0.0, 1.0),
        st.floats(0.001, 0.2),
        st.floats(0.0, 1.0),
        st.floats(0.0, 0.6),
    )
    def test_an_envelope_is_delayed_like_a_note_that_starts_later(
        self, attack, decay, sustain, release, held, wait
    ):
        envelope = adsr_envelope(attack, decay, sustain, release).subs(
            ACTIVE_DURATION, held
        )
        t = np.linspace(0, 1.5, 301)
        assume(np.all(np.abs(t - wait) > 1e-6))
        out = evaluate(delay(envelope, wait), t)
        expected = np.where(
            t > wait, ref_adsr(attack, decay, sustain, release, held, t - wait), 0.0
        )
        np.testing.assert_allclose(out, expected, atol=1e-8)


class TestAlgebra:
    @given(frequencies, seconds, seconds, times)
    def test_delaying_twice_adds_the_delays(self, frequency, first, second, t):
        assume(np.all(np.abs(t - first - second) > 1e-9))
        assume(np.all(np.abs(t - first) > 1e-9) and np.all(np.abs(t - second) > 1e-9))
        twice = evaluate(delay(delay(sine(frequency), first), second), t)
        once = evaluate(delay(sine(frequency), first + second), t)
        np.testing.assert_allclose(twice, once, atol=1e-9)

    @given(frequencies, frequencies, seconds, times)
    def test_it_is_linear(self, a, b, wait, t):
        assume(np.all(np.abs(t - wait) > 1e-9))
        both = evaluate(delay(2 * sine(a) + 3 * sine(b), wait), t)
        apart = 2 * evaluate(delay(sine(a), wait), t) + 3 * evaluate(
            delay(sine(b), wait), t
        )
        np.testing.assert_allclose(both, apart, atol=1e-9)

    def test_it_commutes_with_a_later_shift_of_the_whole_note(self):
        note = beep()
        plain = compile(delay(note, 0.25), 1.0)
        later = compile(delay(note, 0.25).subs(TIME, TIME - 0.1), 1.0)
        np.testing.assert_allclose(later, shifted(plain, 4410), atol=1e-9)

    def test_it_is_a_plain_expression_of_time(self):
        expression = delay(sine(440) * adsr_envelope(0.01, 0.1, 0.5, 0.2), 0.1)
        assert expression.free_symbols == {TIME, ACTIVE_DURATION}
        assert delay(sine(440), 0.1).free_symbols == {TIME}

    def test_a_single_instant_can_be_evaluated(self):
        function = sp.lambdify(TIME, delay(sine(1), 0.5), "numpy")
        assert function(0.1) == 0.0
        assert function(0.75) == pytest.approx(1.0)


class TestInsideAFilter:
    """A filter starts at rest on the first sample, so a delayed filter must be
    the filtered signal moved later, not a filter that had a head start"""

    @pytest.mark.parametrize(
        "signal",
        [
            low_pass_filter(sine(220), 300),
            low_pass_filter(sawtooth(110), 500),
            low_pass_filter(sine(220), 300 + 400 * TIME),
            low_pass_filter(beep(), 800),
            0.5 * sine(110) + low_pass_filter(sawtooth(220), 600),
        ],
        ids=["sine", "sawtooth", "sweeping cutoff", "gated", "mixed"],
    )
    @pytest.mark.parametrize("wait", [0.1, 0.25])
    def test_delaying_a_filter_is_filtering_then_delaying(self, signal, wait):
        count = round(wait * SR)
        np.testing.assert_allclose(
            compile(delay(signal, wait), 1.0),
            shifted(compile(signal, 1.0), count),
            atol=1e-9,
        )

    def test_a_signal_that_starts_away_from_zero_differs_by_the_filters_start_up(self):
        """The filter's first output is always 0, so the filtered signal has
        that start-up built in. A delayed filter starts when its signal does and
        is fed x[0] at once: the two differ by exactly b * x[0] * a**n, the
        filter's own first-sample response, which is gone within a few ms"""
        cutoff, count = 600, 4410
        signal = low_pass_filter(triangle(220), cutoff)
        assert evaluate(triangle(220), [0.0])[0] == 1.0
        late = compile(delay(signal, 0.1), 1.0)
        reference = shifted(compile(signal, 1.0), count)
        k = np.pi * cutoff / SR
        b, a = k / (1 + k), (1 - k) / (1 + k)
        np.testing.assert_allclose(
            (late - reference)[count : count + 200],
            b * a ** np.arange(200),
            atol=1e-9,
        )
        np.testing.assert_allclose((late - reference)[count + 2000 :], 0.0, atol=1e-6)

    def test_a_filter_of_a_delayed_signal_is_fine_too(self):
        signal = low_pass_filter(delay(sine(220), 0.25), 300)
        samples = compile(signal, 1.0)
        assert not samples[:11025].any()
        assert np.abs(samples[11025:]).max() > 0.5

    def test_a_filter_inside_a_filter_inside_a_delay(self):
        signal = low_pass_filter(low_pass_filter(sine(220), 400) + sine(50), 300)
        np.testing.assert_allclose(
            compile(delay(signal, 0.2), 1.0),
            shifted(compile(signal, 1.0), 8820),
            atol=1e-9,
        )

    def test_delaying_twice_through_a_filter(self):
        signal = low_pass_filter(sine(220), 300)
        np.testing.assert_allclose(
            compile(delay(delay(signal, 0.1), 0.15), 1.0),
            shifted(compile(signal, 1.0), 11025),
            atol=1e-9,
        )


class TestSettings:
    @pytest.mark.parametrize("wait", [-0.1, -1, sp.Rational(-1, 3)])
    def test_a_negative_delay_is_refused(self, wait):
        with pytest.raises(ValueError, match="must not be negative"):
            delay(sine(440), wait)

    @pytest.mark.parametrize("wait", [TIME, TIME + 1, sp.Symbol("x"), sp.I])
    def test_a_delay_must_be_a_plain_number(self, wait):
        with pytest.raises(ValueError, match="plain number"):
            delay(sine(440), wait)

    def test_a_signal_may_be_a_plain_number(self):
        samples = compile(delay(sp.Float(1.0), 0.001), 0.01)
        assert samples[44] == 0.0 and samples[45] == 1.0


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="WHITE_NOISE is one shared symbol, so a delayed copy sees the same "
    "noise samples as the original instead of the noise from earlier",
)
def test_delayed_noise_is_the_noise_from_earlier():
    noise = compile(WHITE_NOISE, 0.01)
    delayed = compile(delay(WHITE_NOISE, 0.001), 0.01)
    np.testing.assert_allclose(delayed[44:], noise[:-44], atol=1e-12)


class TestAnEchoBuiltFromDelays:
    def test_each_repeat_is_the_sound_again_quieter(self):
        dry = compile(beep(), 1.0)
        wet = compile(echo(beep(), 0.25, 0.5, 3), 1.0)
        expected = dry.copy()
        for k in (1, 2, 3):
            expected += 0.5**k * shifted(dry, k * 11025)
        np.testing.assert_allclose(wet, expected, atol=1e-9)

    def test_repeats_fade_by_the_feedback(self):
        wet = compile(echo(beep(), 0.25, 0.5, 4), 1.5)
        heights = [np.abs(wet[k * 11025 : k * 11025 + 2205]).max() for k in range(5)]
        np.testing.assert_allclose(
            np.array(heights) / heights[0], [0.5**k for k in range(5)], rtol=1e-6
        )

    def test_a_whole_period_delay_reinforces_a_steady_tone(self):
        # 441 Hz is exactly 100 samples a period at 44100
        wet = compile(echo(sine(441), 100 / SR, 0.5, 20), 1.0)
        assert np.abs(wet[SR // 2 :]).max() == pytest.approx(
            1 + 0.5 / (1 - 0.5), rel=1e-3
        )

    def test_the_repeats_keep_the_pitch(self):
        wet = compile(echo(beep(500, 0.05), 0.25, 0.5, 2), 1.0)
        repeat = wet[11025 : 11025 + 2205]
        spectrum = np.abs(np.fft.rfft(repeat * np.hanning(len(repeat)), 1 << 16))
        assert np.argmax(spectrum) * SR / (1 << 16) == pytest.approx(500, abs=3)

    def test_it_is_made_of_nothing_but_expressions(self):
        expression = echo(beep(), 0.25, 0.5, 3)
        assert expression.free_symbols == {TIME}
        assert expression.has(sp.Piecewise)


class TestAReverbBuiltFromDelays:
    @staticmethod
    def height(samples, start, stop):
        return np.sqrt(np.mean(samples[int(start * SR) : int(stop * SR)] ** 2))

    def test_the_dry_sound_comes_first_and_the_room_answers_after_the_shortest_delay(
        self,
    ):
        samples = compile(reverb(beep(), 1.0, 1.0), 1.5)
        dry = compile(beep(), 1.5)
        first = round(min(COMBS) * SR)
        np.testing.assert_allclose(samples[:first], dry[:first], atol=1e-12)
        assert not np.allclose(samples[first : first + 500], dry[first : first + 500])

    def test_it_rings_on_after_the_sound_stops(self):
        samples = compile(reverb(beep(), 1.0, 1.0), 2.0)
        assert self.height(samples, 0.1, 0.2) > 0.005

    @pytest.mark.parametrize("decay", [0.5, 1.0])
    def test_the_tail_falls_with_the_decay_time(self, decay):
        # 60 dB in `decay` seconds, so 48 dB over the first four fifths of it
        samples = compile(reverb(beep(), decay, 1.0), 2.0 * decay + 0.5)
        early = self.height(samples, 0.1, 0.2)
        later = self.height(samples, 0.1 + 0.8 * decay, 0.2 + 0.8 * decay)
        assert 20 * np.log10(early / later) == pytest.approx(48, abs=10)

    def test_the_tail_stops_where_the_last_copy_falls_60_dB(self):
        samples = compile(reverb(beep(), 1.0, 1.0), 2.0)
        assert self.height(samples, 1.2, 1.5) == 0.0

    def test_it_is_the_sum_of_every_copy(self):
        t = np.arange(SR) / SR
        wet = np.zeros_like(t)
        for spacing in COMBS:
            gain = 10 ** (-3 * spacing / 0.8)
            for k in range(1, int(0.8 / spacing) + 1):
                late = k * spacing
                wet += gain**k * np.where(t >= late, evaluate(beep(), t - late), 0.0)
        expected = evaluate(beep(), t) + 0.3 * wet / math.sqrt(len(COMBS))
        np.testing.assert_allclose(
            compile(reverb(beep(), 0.8, 0.3), 1.0), expected, atol=1e-9
        )

    def test_more_level_means_a_louder_tail(self):
        quiet = compile(reverb(beep(), 1.0, 0.1), 1.5)
        loud = compile(reverb(beep(), 1.0, 0.9), 1.5)
        assert self.height(loud, 0.2, 0.6) == pytest.approx(
            9 * self.height(quiet, 0.2, 0.6), rel=1e-6
        )

    def test_it_is_finite_and_does_not_clip(self):
        samples = compile(reverb(beep(), 1.0, 0.5), 2.0)
        assert np.isfinite(samples).all() and np.abs(samples).max() < 1.5
