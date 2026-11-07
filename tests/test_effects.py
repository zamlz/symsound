"""Echo and reverb are placeholders that are filled in when a signal is compiled.

References, all independent of daw.effects: the echo as a direct sum of
delayed copies (the library runs a recursive filter), and the reverb as a
sample-by-sample loop over ring buffers (the library builds an impulse
response with `lfilter` and convolves). Closed forms cover a sine going
through the echo, and the reverb's decay time is measured on its own tail.
"""

import math

import numpy as np
import pytest
import sympy as sp
from hypothesis import given
from hypothesis import strategies as st
from scipy import signal

from daw.effects import (
    ALLPASS_DELAYS,
    COMB_DELAYS,
    Echo,
    Reverb,
    apply_echo,
    apply_reverb,
    echo,
    reverb,
    reverb_impulse_response,
)
from daw.envelope import ACTIVE_DURATION, adsr_envelope
from daw.waves.basic import sine
from daw.waves.time import TIME
from daw.waves.utils import compile

SR = 44100


def click(length, at=0):
    out = np.zeros(length)
    out[at] = 1.0
    return out


def grid(duration, sample_rate=SR):
    return np.arange(int(duration * sample_rate)) / sample_rate


def burst(frequency=440, length=0.05):
    """A sine that is on for `length` seconds from time zero and silent either side"""
    return sine(frequency) * sp.Piecewise((1, (TIME >= 0) & (TIME < length)), (0, True))


def naive_echo(x, shift, feedback, level):
    """The echo written as the sum of its repeats: repeat k comes k * shift
    samples later at level * feedback**(k - 1)"""
    out = x.copy()
    k = 1
    while k * shift < len(x):
        gain = level * feedback ** (k - 1)
        out[k * shift :] += gain * x[: len(x) - k * shift]
        k += 1
    return out


def naive_reverb_response(decay, damping, sample_rate):
    """Freeverb's comb and all-pass loops, one sample at a time over ring
    buffers, fed a click and scaled to energy 1"""
    length = int(1.5 * decay * sample_rate) + 4096

    def tuned(samples):
        return max(round(samples * sample_rate / 44100), 1)

    def comb(x, samples):
        size = tuned(samples)
        gain = 10 ** (-3 * size / (decay * sample_rate))
        buffer = np.zeros(size)
        store = 0.0
        out = np.zeros(len(x))
        for n, value in enumerate(x):
            i = n % size
            out[n] = buffer[i]
            store = out[n] * (1 - damping) + store * damping
            buffer[i] = value + store * gain
        return out

    def allpass(x, samples):
        size = tuned(samples)
        buffer = np.zeros(size)
        out = np.zeros(len(x))
        for n, value in enumerate(x):
            i = n % size
            held = buffer[i]
            # The textbook all-pass: v[n] = x[n] + g v[n-D], y[n] = -g v[n] + v[n-D]
            stored = value + 0.5 * held
            out[n] = -0.5 * stored + held
            buffer[i] = stored
        return out

    wet = sum(comb(click(length), size) for size in COMB_DELAYS)
    for size in ALLPASS_DELAYS:
        wet = allpass(wet, size)
    return wet / np.sqrt(np.sum(wet**2))


def reverb_time(response, sample_rate, low=-5, high=-35):
    """Seconds to fall 60 dB, from the slope of Schroeder's backward
    integrated energy between two levels"""
    remaining = np.cumsum((response**2)[::-1])[::-1]
    decibels = 10 * np.log10(remaining / remaining[0])
    start = np.argmax(decibels < low)
    stop = np.argmax(decibels < high)
    return (stop - start) / sample_rate * 60 / (low - high)


def band(response, sample_rate, low=None, high=None):
    """The part of a response between two frequencies"""
    if low is None:
        sos = signal.butter(4, high, "low", fs=sample_rate, output="sos")
    elif high is None:
        sos = signal.butter(4, low, "high", fs=sample_rate, output="sos")
    else:
        sos = signal.butter(4, [low, high], "band", fs=sample_rate, output="sos")
    return signal.sosfilt(sos, response)


positions = st.integers(min_value=1, max_value=400)
feedbacks = st.floats(min_value=0.0, max_value=0.95)
levels = st.floats(min_value=0.0, max_value=2.0)
decays = st.floats(min_value=0.05, max_value=0.4)
dampings = st.floats(min_value=0.0, max_value=0.95)
sample_arrays = st.lists(
    st.floats(min_value=-1, max_value=1), min_size=1, max_size=600
).map(np.array)


class TestEchoKnownValues:
    def test_a_click_comes_back_at_each_delay_and_fades_by_the_feedback(self):
        out = apply_echo(click(SR), 0.25, 0.5, 0.8, SR)
        taps = {0: 1.0, 11025: 0.8, 22050: 0.4, 33075: 0.2}
        for position, height in taps.items():
            assert out[position] == pytest.approx(height, abs=1e-12)
        mask = np.ones(SR, dtype=bool)
        mask[list(taps)] = False
        assert np.abs(out[mask]).max() == 0.0

    def test_no_feedback_gives_exactly_one_repeat(self):
        out = apply_echo(click(100), 10 / SR, 0.0, 0.7, SR)
        assert out[0] == 1.0
        assert out[10] == pytest.approx(0.7, abs=1e-15)
        assert np.count_nonzero(out) == 2

    def test_no_level_leaves_the_sound_alone_whatever_the_feedback(self):
        x = np.random.default_rng(1).uniform(-1, 1, 500)
        np.testing.assert_allclose(apply_echo(x, 0.003, 0.9, 0.0, SR), x, atol=1e-12)

    def test_silence_stays_silent(self):
        assert not apply_echo(np.zeros(300), 0.001, 0.5, 0.5, SR).any()

    def test_a_delay_longer_than_the_sound_changes_nothing(self):
        x = np.random.default_rng(2).uniform(-1, 1, 100)
        np.testing.assert_array_equal(apply_echo(x, 1.0, 0.5, 0.5, SR), x)

    def test_the_delay_is_rounded_to_the_nearest_sample(self):
        out = apply_echo(click(40), 10.4 / SR, 0.0, 1.0, SR)
        assert out[10] == pytest.approx(1.0)
        out = apply_echo(click(40), 10.6 / SR, 0.0, 1.0, SR)
        assert out[11] == pytest.approx(1.0)

    def test_the_output_is_as_long_as_the_input(self):
        assert len(apply_echo(np.ones(123), 0.001, 0.5, 0.5, SR)) == 123
        assert len(apply_echo(np.ones(0), 0.001, 0.5, 0.5, SR)) == 0


class TestEchoAgainstTheDirectSum:
    @given(sample_arrays, positions, feedbacks, levels)
    def test_matches_the_sum_of_repeats(self, x, shift, feedback, level):
        out = apply_echo(x, shift / SR, feedback, level, SR)
        np.testing.assert_allclose(
            out, naive_echo(x, shift, feedback, level), atol=1e-9
        )

    @given(sample_arrays, sample_arrays, positions, feedbacks, levels)
    def test_it_is_linear(self, a, b, shift, feedback, level):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        both = apply_echo(a + 2 * b, shift / SR, feedback, level, SR)
        apart = apply_echo(a, shift / SR, feedback, level, SR) + 2 * apply_echo(
            b, shift / SR, feedback, level, SR
        )
        np.testing.assert_allclose(both, apart, atol=1e-9)

    @given(st.integers(0, 200), positions, feedbacks, levels)
    def test_starting_later_only_moves_the_echo_later(self, start, shift, fb, level):
        early = apply_echo(click(700), shift / SR, fb, level, SR)
        late = apply_echo(click(700, at=start), shift / SR, fb, level, SR)
        np.testing.assert_allclose(late[start:], early[: 700 - start], atol=1e-9)
        assert not late[:start].any()

    @given(sample_arrays, positions, feedbacks, levels)
    def test_the_height_never_passes_the_sum_of_the_repeats(
        self, x, shift, feedback, level
    ):
        out = apply_echo(x, shift / SR, feedback, level, SR)
        limit = np.abs(x).max() * (1 + level / (1 - feedback))
        assert np.abs(out).max() <= limit + 1e-9

    @given(sample_arrays, positions, feedbacks, levels)
    def test_echoing_twice_is_echoing_once_with_the_delays_added(
        self, x, shift, feedback, level
    ):
        """Two echoes with no feedback of delay d and e make taps at d, e, d+e"""
        once = apply_echo(x, shift / SR, 0.0, level, SR)
        twice = apply_echo(once, 2 * shift / SR, 0.0, 0.5, SR)
        direct = naive_echo(x, shift, 0.0, level)
        np.testing.assert_allclose(once, direct, atol=1e-9)
        np.testing.assert_allclose(
            twice, naive_echo(direct, 2 * shift, 0.0, 0.5), atol=1e-9
        )


class TestEchoOnASine:
    """A steady sine through the echo has a closed form: the repeats are the
    same sine, delayed, so they add as a geometric series of phasors"""

    @pytest.mark.parametrize("cycles", [1, 3])
    @pytest.mark.parametrize("feedback, level", [(0.5, 0.5), (0.8, 0.3), (0.0, 1.0)])
    def test_a_delay_of_whole_periods_adds_up_in_step(self, cycles, feedback, level):
        frequency = 441  # 100 samples a period at 44100
        shift = 100 * cycles
        t = grid(1.0)
        out = apply_echo(
            np.sin(2 * np.pi * frequency * t), shift / SR, feedback, level, SR
        )
        steady = out[SR // 2 :]
        expected = 1 + level / (1 - feedback)
        assert np.abs(steady).max() == pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize("feedback, level", [(0.5, 0.5), (0.8, 0.3)])
    def test_a_delay_of_half_a_period_is_out_of_step(self, feedback, level):
        t = grid(1.0)
        out = apply_echo(np.sin(2 * np.pi * 441 * t), 50 / SR, feedback, level, SR)
        # Repeat k is shifted by k half periods: it flips sign each time
        expected = 1 - level / (1 + feedback)
        assert np.abs(out[SR // 2 :]).max() == pytest.approx(expected, rel=1e-3)


class TestEchoSettings:
    @pytest.mark.parametrize(
        "delay, feedback, level, message",
        [
            (0.0, 0.5, 0.5, "at least one sample"),
            (-0.1, 0.5, 0.5, "at least one sample"),
            (0.00001, 0.5, 0.5, "at least one sample"),
            (0.1, 1.0, 0.5, "feedback"),
            (0.1, 1.5, 0.5, "feedback"),
            (0.1, -0.1, 0.5, "feedback"),
            (0.1, 0.5, -0.1, "level"),
        ],
    )
    def test_nonsense_is_refused(self, delay, feedback, level, message):
        with pytest.raises(ValueError, match=message):
            apply_echo(np.ones(10), delay, feedback, level, SR)

    def test_a_sample_rate_that_is_not_positive_is_refused(self):
        with pytest.raises(ValueError, match="sample rate"):
            apply_echo(np.ones(10), 0.1, 0.5, 0.5, 0)

    @pytest.mark.parametrize(
        "delay, feedback, level, message",
        [
            (0, 0.5, 0.5, "delay"),
            (-1, 0.5, 0.5, "delay"),
            (0.1, 1, 0.5, "feedback"),
            (0.1, -0.5, 0.5, "feedback"),
            (0.1, 0.5, -1, "level"),
        ],
    )
    def test_the_expression_refuses_them_too(self, delay, feedback, level, message):
        with pytest.raises(ValueError, match=message):
            echo(sine(440), delay, feedback, level)

    @pytest.mark.parametrize("setting", ["delay", "feedback", "level"])
    def test_a_setting_cannot_change_with_time(self, setting):
        values = {"delay": 0.1, "feedback": 0.5, "level": 0.5}
        values[setting] = TIME
        with pytest.raises(ValueError, match=f"{setting} must be a plain number"):
            echo(sine(440), **values)

    def test_a_setting_cannot_be_imaginary(self):
        with pytest.raises(ValueError, match="plain number"):
            echo(sine(440), sp.I)

    def test_settings_may_be_exact_numbers(self):
        expression = echo(sine(440), sp.Rational(1, 4), sp.Rational(1, 2), 1)
        assert isinstance(expression, Echo)


class TestEchoAsAnExpression:
    def test_it_is_a_signal_that_mixes_like_any_other(self):
        expression = echo(sine(440), 0.1)
        assert expression.free_symbols == {TIME}
        assert isinstance(expression, Echo)
        mixed = 0.5 * expression + sine(220)
        samples = compile(mixed, 0.5)
        assert samples.shape == (22050,) and np.isfinite(samples).all()

    def test_compiling_adds_the_echoes_to_the_dry_sound(self):
        dry = burst()
        wet = compile(echo(dry, 0.25, 0.5, 0.8), 1.0)
        plain = compile(dry, 1.0)
        # The sound itself is untouched
        np.testing.assert_allclose(wet[:2205], plain[:2205], atol=1e-12)
        # The repeats are the burst again, 0.8, 0.4, 0.2 times as loud
        for n, height in enumerate([0.8, 0.4, 0.2], start=1):
            start = 11025 * n
            np.testing.assert_allclose(
                wet[start : start + 2205], height * plain[:2205], atol=1e-9
            )
        assert np.abs(wet[2205:11025]).max() == 0.0

    def test_the_repeats_keep_the_pitch_of_the_sound(self):
        samples = compile(echo(burst(500), 0.25, 0.5, 0.8), 0.6)
        repeat = samples[11025 : 11025 + 2205]
        spectrum = np.abs(np.fft.rfft(repeat * np.hanning(len(repeat)), 1 << 16))
        pitch = np.argmax(spectrum) * SR / (1 << 16)
        assert pitch == pytest.approx(500, abs=3)

    @pytest.mark.parametrize("rate", [22050, 44100, 48000])
    def test_the_delay_is_in_seconds_whatever_the_sample_rate(self, rate):
        samples = compile(echo(burst(500, 0.01), 0.2, 0.0, 1.0), 0.5, rate)
        loud = np.flatnonzero(np.abs(samples) > 0.5)
        first_repeat = loud[loud > 0.1 * rate][0] / rate
        assert first_repeat == pytest.approx(0.2, abs=2 / 500)

    def test_an_echo_survives_shifting_the_whole_note_later(self):
        note = burst()
        shifted = echo(note, 0.25, 0.5, 0.8).subs(TIME, TIME - 0.1)
        plain = compile(echo(note, 0.25, 0.5, 0.8), 1.0)
        late = compile(shifted, 1.0)
        start = round(0.1 * SR)
        np.testing.assert_allclose(late[start:], plain[: SR - start], atol=1e-9)
        assert not late[:start].any()

    def test_it_follows_a_note_with_an_envelope(self):
        note = (sine(440) * adsr_envelope(0.01, 0.05, 0.5, 0.1)).subs(
            ACTIVE_DURATION, 0.1
        )
        dry = compile(note, 0.6)
        np.testing.assert_allclose(
            compile(echo(note, 0.2, 0.5, 0.6), 0.6),
            naive_echo(dry, 8820, 0.5, 0.6),
            atol=1e-9,
        )

    def test_a_single_instant_cannot_be_evaluated(self):
        function = sp.lambdify(TIME, echo(sine(440), 0.1), "numpy")
        with pytest.raises(ValueError, match="grid of sample times"):
            function(np.zeros((2, 2)))

    def test_one_sample_comes_back_as_it_was(self):
        function = sp.lambdify(TIME, echo(sine(440) + 1, 0.1), "numpy")
        np.testing.assert_allclose(function(np.array([0.0])), [1.0])

    def test_an_uneven_or_backwards_grid_is_refused(self):
        function = sp.lambdify(TIME, echo(sine(440), 0.1), "numpy")
        with pytest.raises(ValueError, match="uniform, increasing"):
            function(np.array([0.0, 0.1, 0.3]))
        with pytest.raises(ValueError, match="uniform, increasing"):
            function(np.array([0.3, 0.2, 0.1]))

    def test_a_constant_signal_is_repeated_on_every_sample(self):
        samples = compile(echo(sp.Float(1.0), 0.001, 0.0, 1.0), 0.01)
        assert samples[0] == 1.0 and samples[43] == 1.0
        assert samples[44] == pytest.approx(2.0)


class TestReverbResponse:
    def test_it_has_unit_energy_whatever_the_room(self):
        for decay, damping in [(0.2, 0.0), (1.0, 0.5), (2.5, 0.9)]:
            response = reverb_impulse_response(decay, damping, SR)
            assert np.sum(response**2) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("decay, damping", [(0.1, 0.0), (0.2, 0.5), (0.15, 0.9)])
    def test_matches_a_sample_by_sample_loop_of_the_same_design(self, decay, damping):
        rate = 22050
        np.testing.assert_allclose(
            reverb_impulse_response(decay, damping, rate),
            naive_reverb_response(decay, damping, rate),
            atol=1e-9,
        )

    def test_nothing_is_heard_before_the_shortest_comb(self):
        response = reverb_impulse_response(1.0, 0.3, SR)
        first = min(COMB_DELAYS)
        assert not response[:first].any()
        assert response[first] != 0.0

    @pytest.mark.parametrize("decay", [0.5, 1.0, 2.0, 3.5])
    def test_the_tail_falls_60_dB_in_the_decay_time(self, decay):
        response = reverb_impulse_response(decay, 0.0, SR)
        assert reverb_time(response, SR) == pytest.approx(decay, rel=0.06)

    def test_the_decay_time_is_in_seconds_whatever_the_sample_rate(self):
        for rate in (22050, 48000):
            response = reverb_impulse_response(1.5, 0.0, rate)
            assert reverb_time(response, rate) == pytest.approx(1.5, rel=0.06)

    def test_damping_makes_the_highs_die_faster_and_leaves_the_lows(self):
        bright = reverb_impulse_response(2.0, 0.0, SR)
        soft = reverb_impulse_response(2.0, 0.8, SR)
        high_bright = reverb_time(band(bright, SR, low=5000), SR)
        high_soft = reverb_time(band(soft, SR, low=5000), SR)
        low_soft = reverb_time(band(soft, SR, high=300), SR)
        assert high_soft < 0.5 * high_bright
        assert low_soft == pytest.approx(2.0, rel=0.15)

    def test_a_longer_decay_gives_a_longer_response(self):
        assert len(reverb_impulse_response(2.0, 0.5, SR)) > len(
            reverb_impulse_response(1.0, 0.5, SR)
        )

    def test_the_tail_has_fallen_far_below_the_start_where_it_is_cut(self):
        response = reverb_impulse_response(1.0, 0.0, SR)
        loudest = np.abs(response).max()
        assert np.abs(response[-2048:]).max() < 1e-3 * loudest

    @pytest.mark.parametrize(
        "decay, damping, message",
        [
            (0.0, 0.5, "decay"),
            (-1.0, 0.5, "decay"),
            (1.0, -0.1, "damping"),
            (1.0, 1.0, "damping"),
        ],
    )
    def test_nonsense_is_refused(self, decay, damping, message):
        with pytest.raises(ValueError, match=message):
            reverb_impulse_response(decay, damping, SR)

    def test_a_sample_rate_that_is_not_positive_is_refused(self):
        with pytest.raises(ValueError, match="sample rate"):
            reverb_impulse_response(1.0, 0.5, 0)


class TestApplyReverb:
    def test_a_click_comes_back_as_the_click_and_then_the_response(self):
        out = apply_reverb(click(20000), 0.3, 0.5, 0.4, SR)
        response = reverb_impulse_response(0.3, 0.4, SR)
        expected = 0.5 * response[:20000]
        expected[0] += 1.0
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_no_level_leaves_the_sound_alone(self):
        x = np.random.default_rng(3).uniform(-1, 1, 3000)
        np.testing.assert_array_equal(apply_reverb(x, 0.5, 0.0, 0.5, SR), x)

    def test_silence_stays_silent(self):
        assert not apply_reverb(np.zeros(5000), 0.5, 0.5, 0.5, SR).any()

    @given(sample_arrays, sample_arrays, st.floats(0.05, 0.3), levels, dampings)
    def test_it_is_linear(self, a, b, decay, level, damping):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        both = apply_reverb(a + 2 * b, decay, level, damping, 8000)
        apart = apply_reverb(a, decay, level, damping, 8000) + 2 * apply_reverb(
            b, decay, level, damping, 8000
        )
        np.testing.assert_allclose(both, apart, atol=1e-9)

    @given(sample_arrays, decays, levels, dampings)
    def test_the_dry_sound_is_never_changed_only_added_to(
        self, x, decay, level, damping
    ):
        out = apply_reverb(x, decay, level, damping, 8000)
        response = reverb_impulse_response(decay, damping, 8000)
        wet = np.convolve(x, response)[: len(x)]
        np.testing.assert_allclose(out, x + level * wet, atol=1e-9)

    def test_the_wet_part_of_noise_is_as_loud_as_the_noise(self):
        """The response has energy 1, so the wet part of white noise has
        the height of the noise, and `level` scales it exactly"""
        noise = np.random.default_rng(4).uniform(-1, 1, 4 * SR)
        for decay, damping in [(0.3, 0.0), (1.5, 0.5)]:
            out = apply_reverb(noise, decay, 1.0, damping, SR)
            wet = out - noise
            ratio = np.sqrt(np.mean(wet[SR:] ** 2) / np.mean(noise[SR:] ** 2))
            assert ratio == pytest.approx(1.0, rel=0.08)

    def test_the_output_is_as_long_as_the_input(self):
        assert len(apply_reverb(np.ones(77), 0.5, 0.3, 0.5, SR)) == 77
        assert len(apply_reverb(np.ones(500000), 0.5, 0.3, 0.5, SR)) == 500000

    def test_a_negative_level_is_refused(self):
        with pytest.raises(ValueError, match="level"):
            apply_reverb(np.ones(10), 1.0, -0.1, 0.5, SR)


class TestReverbAsAnExpression:
    def test_it_is_a_signal_that_mixes_like_any_other(self):
        expression = reverb(sine(440), 0.5)
        assert expression.free_symbols == {TIME}
        assert isinstance(expression, Reverb)
        samples = compile(0.5 * expression + sine(220), 0.5)
        assert samples.shape == (22050,) and np.isfinite(samples).all()

    def test_a_short_sound_rings_on_after_it_stops_and_dies_away(self):
        samples = compile(reverb(burst(500, 0.05), 1.0, 1.0, 0.0), 3.0)

        def height(start, stop):
            return np.sqrt(np.mean(samples[int(start * SR) : int(stop * SR)] ** 2))

        # Until the shortest comb has come round the room says nothing
        dry = compile(burst(500, 0.05), 3.0)
        first = min(COMB_DELAYS)
        np.testing.assert_allclose(samples[:first], dry[:first], atol=1e-12)
        # After the sound has stopped there is still a tail...
        assert height(0.1, 0.2) > 0.005
        # ... that falls 60 dB (to a thousandth) in one decay time
        fall = 20 * np.log10(height(0.2, 0.3) / height(1.2, 1.3))
        assert fall == pytest.approx(60, abs=12)
        assert height(2.5, 3.0) < height(0.1, 0.2) / 100

    def test_the_tail_is_made_of_the_pitch_that_went_in(self):
        """A reverb of a 500 Hz sine puts the whole wash around 500 Hz"""
        samples = compile(reverb(sine(500), 0.5, 1.0, 0.0), 1.0)
        window = samples[SR // 2 :]
        spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window))))
        pitch = np.argmax(spectrum) * SR / len(window)
        assert pitch == pytest.approx(500, abs=3)

    def test_more_level_means_a_louder_tail(self):
        quiet = compile(reverb(burst(), 1.0, 0.1), 1.5)
        loud = compile(reverb(burst(), 1.0, 0.9), 1.5)
        assert np.abs(loud[8000:]).max() == pytest.approx(
            9 * np.abs(quiet[8000:]).max(), rel=1e-6
        )

    def test_a_reverb_survives_shifting_the_whole_note_later(self):
        shifted = reverb(burst(), 0.5).subs(TIME, TIME - 0.1)
        plain = compile(reverb(burst(), 0.5), 1.0)
        late = compile(shifted, 1.0)
        start = round(0.1 * SR)
        np.testing.assert_allclose(late[start:], plain[: SR - start], atol=1e-9)
        assert np.abs(late[:start]).max() < 1e-12

    def test_effects_can_be_stacked(self):
        both = compile(reverb(echo(burst(), 0.25, 0.5, 0.8), 0.5), 1.0)
        separate = apply_reverb(
            compile(echo(burst(), 0.25, 0.5, 0.8), 1.0), 0.5, 0.3, 0.5, SR
        )
        np.testing.assert_allclose(both, separate, atol=1e-9)

    def test_peak_level_of_a_hit_with_reverb_stays_sane(self):
        samples = compile(reverb(burst(200, 0.1), 2.0), 3.0)
        assert np.isfinite(samples).all()
        assert np.abs(samples).max() < 2.0

    @pytest.mark.parametrize("setting", ["decay", "level", "damping"])
    def test_a_setting_cannot_change_with_time(self, setting):
        values = {"decay": 1.0, "level": 0.3, "damping": 0.5}
        values[setting] = TIME
        with pytest.raises(ValueError, match=f"{setting} must be a plain number"):
            reverb(sine(440), **values)

    @pytest.mark.parametrize(
        "decay, level, damping, message",
        [
            (0, 0.3, 0.5, "decay"),
            (-1, 0.3, 0.5, "decay"),
            (1, -0.1, 0.5, "level"),
            (1, 0.3, -0.1, "damping"),
            (1, 0.3, 1, "damping"),
        ],
    )
    def test_nonsense_is_refused(self, decay, level, damping, message):
        with pytest.raises(ValueError, match=message):
            reverb(sine(440), decay, level, damping)

    def test_a_single_instant_cannot_be_evaluated(self):
        function = sp.lambdify(TIME, reverb(sine(440)), "numpy")
        with pytest.raises(ValueError, match="grid of sample times"):
            function(np.zeros((2, 2)))

    def test_one_sample_comes_back_as_it_was(self):
        function = sp.lambdify(TIME, reverb(sine(440) + 1), "numpy")
        np.testing.assert_allclose(function(np.array([0.0])), [1.0])

    def test_an_uneven_grid_is_refused(self):
        function = sp.lambdify(TIME, reverb(sine(440)), "numpy")
        with pytest.raises(ValueError, match="uniform, increasing"):
            function(np.array([0.0, 0.1, 0.3]))


def test_the_tunings_share_no_common_factor():
    """Echoes of equal lengths would line up and ring at one pitch"""
    delays = COMB_DELAYS + ALLPASS_DELAYS
    assert len(set(delays)) == len(delays)
    assert math.gcd(*COMB_DELAYS) == 1
