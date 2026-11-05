import numpy as np
import pytest
import sympy as sp

from daw.filters import low_pass_filter
from daw.waves.basic import sine
from daw.waves.time import TIME
from helpers import evaluate

T = np.linspace(0.0005, 0.6, 600)


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


class TestStepResponse:
    @pytest.mark.parametrize("cutoff", [5, 50, 440])
    def test_matches_the_rc_charging_curve(self, cutoff):
        out = evaluate(low_pass_filter(sp.Integer(1), cutoff), T)
        np.testing.assert_allclose(out, rc_step(cutoff, T), atol=1e-9)

    def test_settles_to_unity_gain_at_dc(self):
        out = evaluate(low_pass_filter(sp.Integer(1), 100), np.array([1.0, 5.0]))
        np.testing.assert_allclose(out, 1.0, atol=1e-9)

    def test_rises_monotonically_without_overshoot(self):
        out = evaluate(low_pass_filter(sp.Integer(1), 20), T)
        assert np.all(np.diff(out) >= 0)
        assert out.max() <= 1.0 + 1e-12

    def test_time_constant_is_one_over_two_pi_cutoff(self):
        cutoff = 10
        tau = 1 / (2 * np.pi * cutoff)
        at_tau = evaluate(low_pass_filter(sp.Integer(1), cutoff), np.array([tau]))
        assert at_tau[0] == pytest.approx(1 - np.exp(-1), abs=1e-9)


class TestSineResponse:
    @pytest.mark.parametrize(
        ("freq", "cutoff"), [(5, 10), (100, 10), (10, 100), (50, 50)]
    )
    def test_matches_the_closed_form_response(self, freq, cutoff):
        out = evaluate(low_pass_filter(sine(freq), cutoff), T)
        np.testing.assert_allclose(out, rc_sine(freq, cutoff, T), atol=1e-9)

    def test_attenuates_by_one_over_root_two_at_the_cutoff(self):
        freq = cutoff = 50
        # Look at the steady state, well after the transient has died out
        t = np.linspace(0.5, 0.52, 4000)
        out = evaluate(low_pass_filter(sine(freq), cutoff), t)
        assert np.abs(out).max() == pytest.approx(1 / np.sqrt(2), abs=1e-3)

    def test_high_frequencies_are_attenuated_more_than_low_ones(self):
        t = np.linspace(0.5, 0.52, 4000)
        low = evaluate(low_pass_filter(sine(20), 100), t)
        high = evaluate(low_pass_filter(sine(1000), 100), t)
        assert np.abs(low).max() > 0.95
        assert np.abs(high).max() < 0.11

    def test_lowering_the_cutoff_reduces_the_output(self):
        t = np.linspace(0.5, 0.52, 4000)
        loud = evaluate(low_pass_filter(sine(60), 200), t)
        quiet = evaluate(low_pass_filter(sine(60), 20), t)
        assert np.abs(quiet).max() < np.abs(loud).max()

    def test_steady_state_lags_the_input(self):
        # A first-order low-pass delays a sine; a pure gain would not
        freq, cutoff = 50, 50
        t = np.linspace(0.5, 0.52, 4000)
        out = evaluate(low_pass_filter(sine(freq), cutoff), t)
        expected_lag = np.arctan(freq / cutoff)
        assert expected_lag == pytest.approx(np.pi / 4)
        reference = np.sin(2 * np.pi * freq * t - expected_lag) / np.sqrt(2)
        np.testing.assert_allclose(out, reference, atol=1e-3)


class TestLinearity:
    def test_is_linear_in_the_input(self):
        a, b = sine(10), sine(80)
        combined = evaluate(low_pass_filter(2 * a + 3 * b, 40), T)
        separate = 2 * evaluate(low_pass_filter(a, 40), T) + 3 * evaluate(
            low_pass_filter(b, 40), T
        )
        np.testing.assert_allclose(combined, separate, atol=1e-9)

    def test_zero_input_gives_zero_output(self):
        out = low_pass_filter(sp.Integer(0), 100)
        assert out == 0


def test_returns_an_expression_of_time():
    out = low_pass_filter(sine(10), 40)
    assert out.free_symbols <= {TIME}


@pytest.mark.parametrize("cutoff", [10.0, 12.5, sp.Float(50)])
@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="floating point cutoffs crash sympy's inverse Laplace transform; "
    "pass an int or sympy Integer/Rational",
)
def test_float_cutoff_is_supported(cutoff):
    low_pass_filter(sp.Integer(1), cutoff)


def reference_time_varying_filter(t, cutoff, signal):
    """Forward-Euler integration of dy/dt = 2*pi*fc(t) * (x - y)"""
    sr = 1 / (t[1] - t[0])
    y = np.zeros_like(signal)
    for i in range(1, len(t)):
        y[i] = y[i - 1] + (signal[i] - y[i - 1]) * 2 * np.pi * cutoff[i] / sr
    return y


@pytest.mark.parametrize(("depth", "rate"), [(15, 1), (100, 20)])
def test_time_varying_cutoff_approximates_a_numerically_integrated_filter(depth, rate):
    # A time-varying cutoff is not strictly a Laplace-domain filter, so this
    # only holds approximately. The tolerance is a regression guard rather
    # than a claim of exactness.
    sr = 44100
    t = np.arange(int(0.05 * sr)) / sr
    cutoff = 150 + depth * np.sin(2 * np.pi * rate * t)
    reference = reference_time_varying_filter(t, cutoff, np.sin(2 * np.pi * 440 * t))
    out = evaluate(low_pass_filter(sine(440), 150 + depth * sine(rate)), t[1:])
    assert np.abs(out - reference[1:]).max() < 0.1 * np.abs(reference).max()
