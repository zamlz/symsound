"""The low-pass filter is a first-order (RC) filter evaluated on the sample
grid, so it works on any signal: sines, square waves, noise, envelopes,
shifted notes.

Three kinds of reference are used and are independent of symsound.filters:
closed forms (RC step/sine response, noise variance), scipy's continuous-time
simulator and ODE solver, and a plain-Python loop of the discretisation.
"""

import numpy as np
import pytest
import sympy as sp
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy import integrate, signal

from helpers import evaluate
from symsound.envelope import ACTIVE_DURATION, adsr_envelope
from symsound.filters import LowPass, apply_low_pass, low_pass_filter
from symsound.waves.basic import sawtooth, sine, square, triangle
from symsound.waves.time import TIME
from symsound.waves.utils import WHITE_NOISE, compile

SR = 44100


def grid(duration, sample_rate=SR):
    return np.arange(int(duration * sample_rate)) / sample_rate


def rc_step(cutoff, t):
    tau = 1 / (2 * np.pi * cutoff)
    return 1 - np.exp(-t / tau)


def rc_sine(freq, cutoff, t):
    """Response of a first-order low-pass to sin(2*pi*freq*t) switched on at
    t = 0: steady state plus the decaying transient"""
    w = 2 * np.pi * freq
    tau = 1 / (2 * np.pi * cutoff)
    return (np.sin(w * t) - w * tau * np.cos(w * t) + w * tau * np.exp(-t / tau)) / (
        1 + (w * tau) ** 2
    )


def scipy_lowpass(x, cutoff, t):
    """Continuous-time RC filter simulated by scipy, input linearly
    interpolated between samples"""
    tau = 1 / (2 * np.pi * cutoff)
    return signal.lsim(([1.0], [tau, 1.0]), x, t, interp=True)[1]


def naive_lowpass(x, cutoff, sample_rate):
    """Trapezoidal integration of dy/dt = 2*pi*fc*(x - y), one sample at a
    time. The filter must start at rest, with y[0] = 0."""
    dt = 1 / sample_rate
    cutoff = np.broadcast_to(np.asarray(cutoff, dtype=float), x.shape)
    y = np.zeros(len(x))
    for n in range(1, len(x)):
        w_prev, w = 2 * np.pi * cutoff[n - 1], 2 * np.pi * cutoff[n]
        slope_prev = w_prev * (x[n - 1] - y[n - 1])
        y[n] = (y[n - 1] + dt / 2 * (slope_prev + w * x[n])) / (1 + dt / 2 * w)
    return y


def steady_gain(out, freq, sample_rate=SR, start=0.3):
    """Amplitude of a sine at `freq` over a whole number of cycles"""
    first = int(start * sample_rate)
    cycles = max(1, int(0.2 * freq))
    length = int(round(cycles / freq * sample_rate))
    window = out[first : first + length]
    return np.sqrt(2) * np.sqrt((window**2).mean())


class TestStepResponse:
    @pytest.mark.parametrize("cutoff", [5, 50, 440])
    def test_matches_the_rc_charging_curve(self, cutoff):
        t = grid(0.3)
        out = evaluate(low_pass_filter(sp.Integer(1), cutoff), t)
        np.testing.assert_allclose(out, rc_step(cutoff, t), atol=2e-3)

    def test_starts_at_rest(self):
        assert evaluate(low_pass_filter(sp.Integer(1), 100), grid(0.01))[0] == 0.0

    def test_settles_to_unity_gain_at_dc(self):
        out = evaluate(low_pass_filter(sp.Integer(1), 100), grid(0.5))
        assert out[-1] == pytest.approx(1.0, abs=1e-9)

    def test_scales_with_a_constant_input(self):
        t = grid(0.1)
        unit = evaluate(low_pass_filter(sp.Integer(1), 80), t)
        for level in (-2, 0.25, 3):
            out = evaluate(low_pass_filter(sp.Float(level), 80), t)
            np.testing.assert_allclose(out, level * unit, atol=1e-12)

    def test_rises_monotonically_without_overshoot(self):
        out = evaluate(low_pass_filter(sp.Integer(1), 20), grid(0.6))
        assert np.all(np.diff(out) >= 0)
        assert out.max() <= 1.0 + 1e-12

    def test_time_constant_is_one_over_two_pi_cutoff(self):
        cutoff = 10
        t = grid(0.1)
        out = evaluate(low_pass_filter(sp.Integer(1), cutoff), t)
        tau = 1 / (2 * np.pi * cutoff)
        at_tau = out[int(round(tau * SR))]
        assert at_tau == pytest.approx(1 - np.exp(-1), abs=1e-3)

    @pytest.mark.parametrize("cutoff", [10.0, 12.5, sp.Float(50), sp.Rational(25, 2)])
    def test_float_and_exact_cutoffs_agree(self, cutoff):
        t = grid(0.05)
        out = evaluate(low_pass_filter(sp.Integer(1), cutoff), t)
        np.testing.assert_allclose(out, rc_step(float(cutoff), t), atol=2e-3)

    def test_int_and_float_cutoffs_give_the_same_samples(self):
        t = grid(0.05)
        as_int = evaluate(low_pass_filter(sine(300), 50), t)
        as_float = evaluate(low_pass_filter(sine(300), 50.0), t)
        np.testing.assert_allclose(as_int, as_float, atol=1e-12)

    def test_a_constant_input_is_broadcast_to_every_sample(self):
        out = compile(low_pass_filter(sp.Integer(1), 100), 0.05)
        assert out.shape == (int(0.05 * SR),)


class TestSineResponse:
    @pytest.mark.parametrize(
        ("freq", "cutoff"), [(5, 10), (100, 10), (10, 100), (50, 50), (440, 100)]
    )
    def test_matches_the_closed_form_response(self, freq, cutoff):
        t = grid(0.3)
        out = evaluate(low_pass_filter(sine(freq), cutoff), t)
        np.testing.assert_allclose(out, rc_sine(freq, cutoff, t), atol=1e-3)

    def test_attenuates_by_one_over_root_two_at_the_cutoff(self):
        out = evaluate(low_pass_filter(sine(50), 50), grid(0.6))
        assert steady_gain(out, 50) == pytest.approx(1 / np.sqrt(2), abs=1e-3)

    @pytest.mark.parametrize("freq", [10, 50, 100, 300, 1000])
    def test_gain_follows_the_one_pole_magnitude_response(self, freq):
        cutoff = 100
        out = evaluate(low_pass_filter(sine(freq), cutoff), grid(0.6))
        expected = 1 / np.sqrt(1 + (freq / cutoff) ** 2)
        assert steady_gain(out, freq) == pytest.approx(expected, rel=1e-2)

    @pytest.mark.parametrize("freq", [100, 1000, 3000, 8000])
    def test_gain_matches_the_exact_response_of_the_discretisation(self, freq):
        # The trapezoidal rule maps frequency f to tan(pi f / fs) * fs / pi,
        # so high frequencies are attenuated slightly more than the analog
        # filter would
        cutoff = 100
        out = evaluate(low_pass_filter(sine(freq), cutoff), grid(0.6))
        warped = SR * np.tan(np.pi * freq / SR) / np.pi
        expected = 1 / np.sqrt(1 + (warped / cutoff) ** 2)
        assert steady_gain(out, freq) == pytest.approx(expected, rel=1e-3)

    def test_rolls_off_at_20_db_per_decade(self):
        cutoff = 20
        gain = {}
        for freq in (200, 2000):
            out = evaluate(low_pass_filter(sine(freq), cutoff), grid(0.6))
            gain[freq] = steady_gain(out, freq)
        assert 20 * np.log10(gain[200] / gain[2000]) == pytest.approx(20, abs=0.3)

    def test_steady_state_lags_the_input(self):
        # A first-order low-pass delays a sine; a pure gain would not
        freq = cutoff = 50
        t = grid(0.6)
        out = evaluate(low_pass_filter(sine(freq), cutoff), t)
        lag = np.arctan(freq / cutoff)
        assert lag == pytest.approx(np.pi / 4)
        reference = np.sin(2 * np.pi * freq * t - lag) / np.sqrt(2)
        np.testing.assert_allclose(
            out[int(0.3 * SR) :], reference[int(0.3 * SR) :], atol=2e-3
        )

    def test_lowering_the_cutoff_reduces_the_output(self):
        t = grid(0.6)
        loud = evaluate(low_pass_filter(sine(60), 200), t)
        quiet = evaluate(low_pass_filter(sine(60), 20), t)
        assert steady_gain(quiet, 60) < steady_gain(loud, 60)


# Signals the Laplace based filter could not handle: sympy returned an
# unevaluated transform, crashed, or (for noise) a wrong answer
ENVELOPE = adsr_envelope(0.05, 0.05, 0.6, 0.1).subs(ACTIVE_DURATION, 0.2)
SIGNALS = {
    "square": (square(110), 2e-2),
    "triangle": (triangle(110), 5e-3),
    "sawtooth": (sawtooth(110), 2e-2),
    "sum of sines": (sine(110) + 0.5 * sine(1000), 5e-3),
    "enveloped sine": (sine(220) * ENVELOPE, 5e-3),
    "enveloped square": (square(220) * ENVELOPE, 2e-2),
    "frequency modulated": (sine(220 + 40 * sine(5)), 5e-3),
    "constant": (sp.Float(0.5), 1e-3),
}


class TestArbitrarySignals:
    @pytest.mark.parametrize("name", SIGNALS)
    def test_matches_a_continuous_time_simulation(self, name):
        expr, atol = SIGNALS[name]
        t = grid(0.3)
        out = evaluate(low_pass_filter(expr, 200), t)
        reference = scipy_lowpass(evaluate(expr, t), 200, t)
        np.testing.assert_allclose(out, reference, atol=atol)

    @pytest.mark.parametrize("name", SIGNALS)
    def test_result_is_a_plain_finite_signal(self, name):
        expr, _ = SIGNALS[name]
        out = compile(low_pass_filter(expr, 200), 0.2)
        assert out.shape == (int(0.2 * SR),)
        assert np.isfinite(out).all()

    def test_smooths_a_square_wave_towards_a_sine(self):
        t = grid(0.6)
        out = evaluate(low_pass_filter(square(100), 100), t)
        steady = out[int(0.3 * SR) :]
        # Exact periodic steady state of an RC filter driven by a square wave
        # peaks at tanh(T / (4 tau)) = tanh(pi * cutoff / (2 * freq))
        assert np.abs(steady).max() == pytest.approx(np.tanh(np.pi / 2), rel=1e-3)
        # The corners are rounded
        assert np.abs(np.diff(steady)).max() < 0.1
        # A symmetric wave through a linear filter stays centred
        assert abs(steady.mean()) < 1e-2

    def test_a_high_cutoff_leaves_a_slow_signal_almost_unchanged(self):
        t = grid(0.2)
        out = evaluate(low_pass_filter(sine(20), 10000), t)
        np.testing.assert_allclose(out[200:], evaluate(sine(20), t)[200:], atol=5e-3)


class TestNoise:
    @staticmethod
    def noise(seconds, seed=0):
        return np.random.default_rng(seed).uniform(-1, 1, int(seconds * SR))

    def test_filters_the_noise_samples(self):
        # Regression: the Laplace filter treated the noise symbol as a constant
        # and returned noise * (1 - exp(-t/tau)), a fade-in of unfiltered noise
        x = self.noise(0.05)
        out = evaluate(low_pass_filter(WHITE_NOISE, 1000), grid(0.05), noise=x)
        np.testing.assert_allclose(out, naive_lowpass(x, 1000, SR), atol=1e-12)
        assert np.corrcoef(out, x)[0, 1] < 0.5  # it is not the input

    @pytest.mark.parametrize("cutoff", [500, 1000, 4000])
    def test_removes_the_expected_share_of_the_noise_power(self, cutoff):
        # Output variance of the one-pole filter on white noise is
        # sigma^2 * k / (1 + k) with k = pi * cutoff / sample_rate
        x = self.noise(3)
        out = evaluate(low_pass_filter(WHITE_NOISE, cutoff), grid(3), noise=x)
        k = np.pi * cutoff / SR
        expected = x.var() * k / (1 + k)
        assert out[SR // 10 :].var() == pytest.approx(expected, rel=0.15)
        assert abs(out.mean()) < 0.01

    def test_lower_cutoffs_leave_less_noise(self):
        x = self.noise(1)
        t = grid(1)
        rms = [
            evaluate(low_pass_filter(WHITE_NOISE, fc), t, noise=x).std()
            for fc in (200, 2000, 10000)
        ]
        assert rms[0] < rms[1] < rms[2] < x.std()

    def test_noise_through_compile_is_filtered(self):
        out = compile(low_pass_filter(WHITE_NOISE, 500), 1.0)
        assert out.std() < 0.2 * np.sqrt(1 / 3)


class TestLinearity:
    def test_is_linear_in_the_input(self):
        a, b = sine(10), square(80)
        t = grid(0.1)
        combined = evaluate(low_pass_filter(2 * a + 3 * b, 40), t)
        separate = 2 * evaluate(low_pass_filter(a, 40), t) + 3 * evaluate(
            low_pass_filter(b, 40), t
        )
        np.testing.assert_allclose(combined, separate, atol=1e-9)

    def test_zero_input_gives_zero_output(self):
        out = evaluate(low_pass_filter(sp.Integer(0), 100), grid(0.01))
        assert not out.any()

    def test_two_filters_in_a_row_make_a_second_order_response(self):
        cutoff = 80
        t = grid(0.3)
        twice = low_pass_filter(low_pass_filter(sp.Integer(1), cutoff), cutoff)
        out = evaluate(twice, t)
        tau = 1 / (2 * np.pi * cutoff)
        expected = 1 - (1 + t / tau) * np.exp(-t / tau)
        np.testing.assert_allclose(out, expected, atol=3e-3)

    def test_filtering_commutes_with_scaling_by_a_constant(self):
        t = grid(0.05)
        scaled_after = 0.3 * evaluate(low_pass_filter(square(90), 60), t)
        scaled_before = evaluate(low_pass_filter(0.3 * square(90), 60), t)
        np.testing.assert_allclose(scaled_after, scaled_before, atol=1e-12)


class TestTimeShifting:
    """Songs are built from notes shifted with subs(TIME, TIME - start)"""

    START = 3000 / SR  # a whole number of samples

    def test_a_filtered_note_shifted_later_is_the_same_note_later(self):
        t = grid(0.4)
        note = sine(330) * ENVELOPE
        plain = evaluate(low_pass_filter(note, 200), t)
        shifted = evaluate(low_pass_filter(note.subs(TIME, TIME - self.START), 200), t)
        n = round(self.START * SR)
        np.testing.assert_allclose(shifted[:n], 0.0, atol=1e-12)
        np.testing.assert_allclose(shifted[n:], plain[:-n], atol=1e-9)

    def test_shifting_the_filtered_expression_is_equivalent(self):
        t = grid(0.4)
        note = sine(330) * ENVELOPE
        inside = low_pass_filter(note.subs(TIME, TIME - self.START), 200)
        outside = low_pass_filter(note, 200).subs(TIME, TIME - self.START)
        np.testing.assert_allclose(evaluate(inside, t), evaluate(outside, t), atol=1e-9)

    def test_shifted_note_is_silent_before_it_starts(self):
        t = grid(0.4)
        note = (sine(330) * ENVELOPE).subs(TIME, TIME - self.START)
        out = evaluate(low_pass_filter(note, 200), t)
        assert not out[: round(self.START * SR)].any()

    def test_many_shifted_notes_summed_then_filtered_stay_bounded(self):
        song = sum(
            (sine(220 + 20 * i) * ENVELOPE).subs(TIME, TIME - 0.05 * i)
            for i in range(6)
        )
        out = evaluate(low_pass_filter(song, 400), grid(0.6))
        assert np.isfinite(out).all()
        assert np.abs(out).max() < 6


class TestTimeVaryingCutoff:
    def test_matches_an_ode_solver_with_the_exact_input(self):
        # dy/dt = 2*pi*fc(t)*(x(t) - y), solved by scipy with no sampling
        def cutoff(t):
            return 150 + 100 * np.sin(2 * np.pi * 20 * t)

        def rhs(t, y):
            return 2 * np.pi * cutoff(t) * (np.sin(2 * np.pi * 440 * t) - y)

        t = grid(0.05)
        solved = integrate.solve_ivp(
            rhs, (0, t[-1]), [0.0], t_eval=t, rtol=1e-10, atol=1e-12
        ).y[0]
        out = evaluate(low_pass_filter(sine(440), 150 + 100 * sine(20)), t)
        np.testing.assert_allclose(out, solved, atol=3e-3)

    def test_a_constant_cutoff_written_as_an_expression_matches_a_number(self):
        t = grid(0.05)
        as_expr = evaluate(low_pass_filter(sine(300), 100 + 0 * sine(7)), t)
        as_number = evaluate(low_pass_filter(sine(300), 100), t)
        np.testing.assert_allclose(as_expr, as_number, atol=1e-12)

    def test_a_cutoff_that_rises_lets_more_signal_through(self):
        t = grid(0.5)
        out = evaluate(low_pass_filter(sine(1000), 20 + 4000 * t), t)
        # the cutoff sweeps from 100 Hz at 20 ms to about 2 kHz at the end
        early = np.abs(out[SR // 50 : SR // 25]).max()
        late = np.abs(out[-SR // 10 :]).max()
        assert early < 0.25
        assert late > 0.8

    def test_the_cutoff_may_depend_on_noise(self):
        x = np.random.default_rng(1).uniform(-1, 1, 500)
        out = evaluate(
            low_pass_filter(sp.Integer(1), 100 + 50 * WHITE_NOISE),
            grid(500 / SR),
            noise=x,
        )
        np.testing.assert_allclose(
            out, naive_lowpass(np.ones(500), 100 + 50 * x, SR), atol=1e-12
        )


class TestSampleRate:
    @pytest.mark.parametrize("rate", [8000, 22050, 48000, 96000])
    def test_step_response_does_not_depend_on_the_sample_rate(self, rate):
        t = grid(0.1, rate)
        out = evaluate(low_pass_filter(sp.Integer(1), 60), t)
        np.testing.assert_allclose(out, rc_step(60, t), atol=3e-3)

    def test_error_shrinks_about_four_times_when_the_rate_doubles(self):
        # Trapezoidal integration is second order
        errors = []
        for rate in (11025, 22050, 44100):
            t = grid(0.05, rate)
            out = evaluate(low_pass_filter(sine(200), 300), t)
            errors.append(np.abs(out - rc_sine(200, 300, t)).max())
        assert errors[0] / errors[1] == pytest.approx(4, rel=0.25)
        assert errors[1] / errors[2] == pytest.approx(4, rel=0.25)

    def test_unusual_sample_rates_work_through_compile(self):
        out = compile(low_pass_filter(sine(100), 50), 0.1, 12345)
        assert out.shape == (1234,)
        assert np.isfinite(out).all()


class TestValidation:
    def test_a_negative_cutoff_is_rejected_when_building(self):
        with pytest.raises(ValueError, match="cutoff"):
            low_pass_filter(sine(10), -5)

    def test_a_cutoff_that_goes_negative_is_rejected_when_evaluated(self):
        expr = low_pass_filter(sine(10), 100 * sine(3))
        with pytest.raises(ValueError, match="cutoff"):
            evaluate(expr, grid(0.5))

    def test_a_zero_cutoff_blocks_everything(self):
        assert not evaluate(low_pass_filter(sine(100), 0), grid(0.05)).any()

    def test_a_single_time_cannot_be_filtered(self):
        fn = sp.lambdify((TIME, WHITE_NOISE), low_pass_filter(sine(10), 50), "numpy")
        with pytest.raises(ValueError, match="grid"):
            fn(0.5, 0.0)

    def test_an_uneven_grid_is_rejected(self):
        with pytest.raises(ValueError, match="uniform"):
            evaluate(low_pass_filter(sine(10), 50), np.array([0.0, 0.1, 0.3, 0.4]))

    def test_one_sample_comes_out_at_rest(self):
        out = evaluate(low_pass_filter(sine(10), 50), np.array([0.25]))
        np.testing.assert_array_equal(out, [0.0])

    def test_no_samples_in_no_samples_out(self):
        assert evaluate(low_pass_filter(sine(10), 50), np.array([])).shape == (0,)


class TestExpressionStructure:
    def test_returns_a_sympy_expression_of_time(self):
        out = low_pass_filter(sine(10), 40)
        assert isinstance(out, sp.Expr)
        assert out.free_symbols == {TIME}

    def test_keeps_noise_and_cutoff_symbols_visible(self):
        out = low_pass_filter(WHITE_NOISE * sine(3), 40 + 10 * sine(2))
        assert out.free_symbols == {TIME, WHITE_NOISE}

    def test_is_a_lowpass_node_that_composes_with_arithmetic(self):
        out = low_pass_filter(sine(10), 40)
        assert isinstance(out, LowPass)
        combined = 0.5 * out + sine(100)
        assert isinstance(combined, sp.Expr)
        assert out.has(LowPass)

    def test_equal_inputs_give_equal_expressions(self):
        assert low_pass_filter(sine(10), 40) == low_pass_filter(sine(10), 40)
        assert low_pass_filter(sine(10), 40) != low_pass_filter(sine(10), 41)

    def test_does_not_use_the_laplace_transform(self):
        out = low_pass_filter(square(100), 50)
        assert not out.has(sp.InverseLaplaceTransform)
        assert not out.has(sp.LaplaceTransform)


class TestApplyLowPass:
    """The sample-domain primitive behind LowPass"""

    def test_output_has_the_input_shape_and_starts_at_rest(self):
        x = np.random.default_rng(0).uniform(-1, 1, 100)
        out = apply_low_pass(x, 500.0, SR)
        assert out.shape == x.shape
        assert out[0] == 0.0

    def test_does_not_modify_its_input(self):
        x = np.random.default_rng(0).uniform(-1, 1, 50)
        before = x.copy()
        apply_low_pass(x, 500.0, SR)
        np.testing.assert_array_equal(x, before)

    def test_a_scalar_input_is_broadcast_against_the_cutoff_array(self):
        out = apply_low_pass(np.ones(40), np.full(40, 300.0), SR)
        np.testing.assert_allclose(
            out, apply_low_pass(np.ones(40), 300.0, SR), atol=1e-12
        )

    def test_a_cutoff_array_of_the_wrong_length_is_rejected(self):
        with pytest.raises(ValueError, match="cutoff"):
            apply_low_pass(np.ones(10), np.ones(7), SR)

    @pytest.mark.parametrize("rate", [0, -44100])
    def test_a_non_positive_sample_rate_is_rejected(self, rate):
        with pytest.raises(ValueError, match="sample rate"):
            apply_low_pass(np.ones(10), 100.0, rate)

    def test_negative_cutoffs_are_rejected(self):
        with pytest.raises(ValueError, match="cutoff"):
            apply_low_pass(np.ones(10), -1.0, SR)
        with pytest.raises(ValueError, match="cutoff"):
            apply_low_pass(np.ones(10), np.array([1.0] * 9 + [-1.0]), SR)

    @settings(max_examples=60)
    @given(
        st.lists(st.floats(-1, 1), min_size=1, max_size=120),
        st.floats(0, 5000),
    )
    def test_matches_the_plain_python_loop(self, xs, cutoff):
        x = np.array(xs)
        np.testing.assert_allclose(
            apply_low_pass(x, cutoff, SR), naive_lowpass(x, cutoff, SR), atol=1e-12
        )

    @settings(max_examples=60)
    @given(st.data(), st.integers(1, 120))
    def test_time_varying_cutoff_matches_the_plain_python_loop(self, data, n):
        x = np.array(data.draw(st.lists(st.floats(-1, 1), min_size=n, max_size=n)))
        cutoff = np.array(
            data.draw(st.lists(st.floats(0, 5000), min_size=n, max_size=n))
        )
        np.testing.assert_allclose(
            apply_low_pass(x, cutoff, SR), naive_lowpass(x, cutoff, SR), atol=1e-12
        )

    @settings(max_examples=60)
    @given(st.lists(st.floats(-1, 1), min_size=2, max_size=120), st.floats(1, 5000))
    def test_a_constant_cutoff_array_equals_a_scalar_cutoff(self, xs, cutoff):
        x = np.array(xs)
        np.testing.assert_allclose(
            apply_low_pass(x, np.full(len(x), cutoff), SR),
            apply_low_pass(x, cutoff, SR),
            atol=1e-12,
        )

    @settings(max_examples=60)
    @given(
        st.lists(st.floats(-1, 1), min_size=1, max_size=80),
        st.lists(st.floats(-1, 1), min_size=1, max_size=80),
        st.floats(-3, 3),
        st.floats(-3, 3),
        st.floats(1, 5000),
    )
    def test_is_linear(self, xs, ys, a, b, cutoff):
        n = min(len(xs), len(ys))
        x, y = np.array(xs[:n]), np.array(ys[:n])
        np.testing.assert_allclose(
            apply_low_pass(a * x + b * y, cutoff, SR),
            a * apply_low_pass(x, cutoff, SR) + b * apply_low_pass(y, cutoff, SR),
            atol=1e-9,
        )

    @settings(max_examples=60)
    @given(
        st.lists(st.floats(-1, 1), min_size=1, max_size=80),
        st.integers(0, 40),
        st.floats(1, 5000),
    )
    def test_delaying_the_input_delays_the_output(self, xs, delay, cutoff):
        # The filter starts at rest, so the signal must too: a signal that is
        # already non-zero at the first sample is a step, which is not shift
        # invariant
        x = np.array([0.0, *xs])
        delayed = np.concatenate([np.zeros(delay), x])
        np.testing.assert_allclose(
            apply_low_pass(delayed, cutoff, SR)[delay:],
            apply_low_pass(x, cutoff, SR),
            atol=1e-12,
        )

    @settings(max_examples=60)
    @given(
        st.lists(st.floats(-1, 1), min_size=1, max_size=200),
        st.floats(1, 10000),
    )
    def test_never_exceeds_the_largest_input(self, xs, cutoff):
        # Below sample_rate / pi the impulse response is non-negative and sums
        # to at most one, so the filter cannot create peaks
        assume(cutoff <= SR / np.pi)
        out = apply_low_pass(np.array(xs), cutoff, SR)
        assert np.abs(out).max() <= np.abs(xs).max() + 1e-9

    @settings(max_examples=40)
    @given(st.lists(st.floats(-1, 1), min_size=1, max_size=200), st.floats(0, 1e7))
    def test_stays_finite_for_any_cutoff_even_far_above_nyquist(self, xs, cutoff):
        out = apply_low_pass(np.array(xs), cutoff, SR)
        assert np.isfinite(out).all()
        assert np.abs(out).max() <= 2 * np.abs(xs).max() + 1e-9

    def test_an_empty_signal_gives_an_empty_result(self):
        assert apply_low_pass(np.array([]), 100.0, SR).shape == (0,)
