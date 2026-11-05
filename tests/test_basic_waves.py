import numpy as np
import pytest
import sympy as sp
from hypothesis import assume, given
from hypothesis import strategies as st

from daw.constants import Note
from daw.waves.basic import sawtooth, sawtooth_v2, sine, square, triangle, trigger
from daw.waves.time import TIME
from helpers import (
    away_from_edges,
    cycle_fraction,
    evaluate,
    ref_sawtooth,
    ref_square,
    ref_triangle,
)

frequencies = st.floats(min_value=0.5, max_value=4000, allow_nan=False)
times = st.lists(
    st.floats(min_value=0, max_value=2, allow_nan=False), min_size=1, max_size=12
)

# (constructor, independent reference implementation)
WAVES = {
    "sine": (sine, lambda f, t: np.sin(2 * np.pi * f * np.asarray(t))),
    "square": (square, ref_square),
    "triangle": (triangle, ref_triangle),
    "sawtooth": (sawtooth, ref_sawtooth),
}
wave_param = pytest.mark.parametrize("name", WAVES)


def smooth_wave(name):
    return name == "sine"


@wave_param
@given(frequencies, times)
def test_matches_reference_implementation(name, f, t):
    make, ref = WAVES[name]
    t = np.array(t)
    if not smooth_wave(name):
        keep = away_from_edges(f, t)
        assume(keep.any())
        t = t[keep]
    np.testing.assert_allclose(evaluate(make(f), t), ref(f, t), atol=1e-9)


@wave_param
@given(frequencies)
def test_output_stays_within_unit_range(name, f):
    make, _ = WAVES[name]
    t = np.linspace(0, 3 / f, 997)
    values = evaluate(make(f), t)
    assert np.isfinite(values).all()
    assert np.abs(values).max() <= 1 + 1e-12


@wave_param
@given(frequencies, times, st.integers(min_value=1, max_value=4))
def test_is_periodic(name, f, t, k):
    make, _ = WAVES[name]
    t = np.array(t)
    keep = away_from_edges(f, t, margin=1e-5)
    assume(keep.any())
    t = t[keep]
    wave = make(f)
    np.testing.assert_allclose(evaluate(wave, t + k / f), evaluate(wave, t), atol=1e-6)


@wave_param
@given(frequencies, times)
def test_doubling_frequency_doubles_the_rate(name, f, t):
    make, _ = WAVES[name]
    t = np.array(t)
    keep = away_from_edges(f, 2 * t, margin=1e-5)
    assume(keep.any())
    t = t[keep]
    np.testing.assert_allclose(
        evaluate(make(2 * f), t), evaluate(make(f), 2 * t), atol=1e-6
    )


@wave_param
@given(
    frequencies,
    times,
    st.floats(min_value=-1, max_value=1, allow_nan=False),
)
def test_time_shift_is_equivalent_to_evaluating_later(name, f, t, shift):
    make, _ = WAVES[name]
    t = np.array(t)
    keep = away_from_edges(f, t + shift, margin=1e-5)
    assume(keep.any())
    t = t[keep]
    shifted = make(f, time_shift=TIME + shift)
    np.testing.assert_allclose(
        evaluate(shifted, t), evaluate(make(f), t + shift), atol=1e-6
    )


@wave_param
def test_default_frequency_is_concert_a(name):
    make, _ = WAVES[name]
    t = np.linspace(0, 0.01, 101)
    np.testing.assert_allclose(
        evaluate(make(), t), evaluate(make(Note.A4), t), atol=1e-12
    )


@wave_param
@given(st.integers(min_value=1, max_value=200))
def test_has_zero_mean_over_whole_periods(name, f):
    make, _ = WAVES[name]
    n = 1000
    # Sample at the midpoint of each step so no sample sits on a crossing
    t = (np.arange(n) + 0.5) / (n * f)
    assert evaluate(make(f), t).mean() == pytest.approx(0, abs=1e-9)


@wave_param
def test_accepts_a_numpy_float_frequency(name):
    make, _ = WAVES[name]
    wave = make(np.float64(440))
    assert wave.free_symbols == {TIME}


@wave_param
def test_expression_depends_only_on_time(name):
    make, _ = WAVES[name]
    assert make(440).free_symbols == {TIME}


class TestSine:
    def test_key_points_of_one_hertz_sine(self):
        t = np.array([0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(evaluate(sine(1), t), [0, 1, 0, -1, 0], atol=1e-12)

    @given(frequencies, times)
    def test_is_an_odd_function_of_time(self, f, t):
        t = np.array(t)
        np.testing.assert_allclose(
            evaluate(sine(f), -t), -evaluate(sine(f), t), atol=1e-9
        )

    @given(
        st.floats(min_value=20, max_value=2000, allow_nan=False),
        st.floats(min_value=0.1, max_value=50, allow_nan=False),
        st.floats(min_value=0.5, max_value=20, allow_nan=False),
        times,
    )
    def test_frequency_modulation_integrates_the_frequency(self, f0, depth, fm, t):
        # The phase of sin(2*pi*integral(f)) with f = f0 + depth*sin(2*pi*fm*t)
        # is f0*t - depth*cos(2*pi*fm*t) / (2*pi*fm) (zero constant of
        # integration), which makes the instantaneous frequency equal f(t).
        t = np.array(t)
        wave = sine(f0 + depth * sine(fm))
        phase = f0 * t - depth * np.cos(2 * np.pi * fm * t) / (2 * np.pi * fm)
        np.testing.assert_allclose(
            evaluate(wave, t), np.sin(2 * np.pi * phase), atol=1e-8
        )

    def test_instantaneous_frequency_follows_the_modulator(self):
        sr = 200_000
        t = np.arange(int(0.2 * sr)) / sr
        f0, depth, fm = 1000.0, 200.0, 5.0
        wave = evaluate(sine(f0 + depth * sine(fm)), t)

        # Count rising zero crossings in a window around two modulator phases
        def local_frequency(center):
            lo, hi = int((center - 0.005) * sr), int((center + 0.005) * sr)
            seg = wave[lo:hi]
            crossings = np.sum((seg[:-1] < 0) & (seg[1:] >= 0))
            return crossings / 0.01

        peak_time = 1 / (4 * fm)  # sine modulator is +1 here
        trough_time = 3 / (4 * fm)  # and -1 here
        assert local_frequency(peak_time) == pytest.approx(f0 + depth, abs=100)
        assert local_frequency(trough_time) == pytest.approx(f0 - depth, abs=100)

    def test_amplitude_scaling_is_linear(self):
        t = np.linspace(0, 0.01, 50)
        np.testing.assert_allclose(
            evaluate(0.25 * sine(300), t), 0.25 * evaluate(sine(300), t)
        )


class TestSquare:
    @given(frequencies, times)
    def test_only_takes_the_values_plus_and_minus_one(self, f, t):
        t = np.array(t)
        keep = away_from_edges(f, t)
        assume(keep.any())
        assert set(np.unique(evaluate(square(f), t[keep]))) <= {-1.0, 1.0}

    def test_positive_first_half_negative_second_half(self):
        t = np.array([0.1, 0.4, 0.6, 0.9, 1.1, 1.6])
        np.testing.assert_array_equal(evaluate(square(1), t), [1, 1, -1, -1, 1, -1])

    def test_is_zero_exactly_at_the_zero_crossing(self):
        assert float(square(1).subs(TIME, 0)) == 0.0


class TestTriangle:
    def test_key_points(self):
        t = np.array([0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(
            evaluate(triangle(1), t), [1, 0, -1, 0, 1], atol=1e-12
        )

    @given(frequencies, times)
    def test_slope_is_constant_in_magnitude(self, f, t):
        # Between kinks the triangle moves 4*f per second
        t = np.array(t)
        h = 1e-7 / f
        keep = away_from_edges(f, t, margin=1e-4) & away_from_edges(
            f, t + h, margin=1e-4
        )
        assume(keep.any())
        t = t[keep]
        slope = (evaluate(triangle(f), t + h) - evaluate(triangle(f), t)) / h
        np.testing.assert_allclose(np.abs(slope), 4 * f, rtol=1e-3)

    @given(frequencies)
    def test_is_continuous(self, f):
        t = np.linspace(0, 3 / f, 50_001)
        steps = np.abs(np.diff(evaluate(triangle(f), t)))
        assert steps.max() <= 4 * f * (t[1] - t[0]) * (1 + 1e-6)


class TestSawtooth:
    def test_key_points(self):
        t = np.array([0.0, 0.25, 0.4999, 0.5001, 0.75, 0.9999])
        np.testing.assert_allclose(
            evaluate(sawtooth(1), t),
            [0, 0.5, 0.9998, -0.9998, -0.5, -0.0002],
            atol=1e-9,
        )

    @given(frequencies, times)
    def test_slope_is_two_per_period(self, f, t):
        t = np.array(t)
        h = 1e-7 / f
        keep = away_from_edges(f, t, margin=1e-4) & away_from_edges(
            f, t + h, margin=1e-4
        )
        assume(keep.any())
        t = t[keep]
        slope = (evaluate(sawtooth(f), t + h) - evaluate(sawtooth(f), t)) / h
        np.testing.assert_allclose(slope, 2 * f, rtol=1e-3)

    def test_jumps_by_two_at_half_a_period(self):
        eps = 1e-9
        before = float(sawtooth(1).subs(TIME, 0.5 - eps))
        after = float(sawtooth(1).subs(TIME, 0.5 + eps))
        assert before - after == pytest.approx(2, abs=1e-6)

    def test_shifted_inverted_sawtooth_is_a_falling_ramp(self):
        # kick.py builds (1 - sawtooth(time_shift=TIME + half a period)) / 2
        # as a ramp that falls monotonically from 1 to 0 over each period
        wave = (1 - sawtooth(1, time_shift=TIME + 0.5)) / 2
        t = np.linspace(0.0001, 0.9999, 500)
        ramp = evaluate(wave, t)
        assert np.all(np.diff(ramp) < 0)
        assert ramp[0] == pytest.approx(1, abs=1e-3)
        assert ramp[-1] == pytest.approx(0, abs=1e-3)


class TestSawtoothV2:
    """The additive-synthesis sawtooth: sum of (-1)^(i+1)/i * sin(2*pi*i*f*t)"""

    @staticmethod
    def reference(f, t, resolution):
        i = np.arange(1, resolution + 1)[:, None]
        terms = ((-1.0) ** (i + 1) / i) * np.sin(2 * np.pi * i * f * np.asarray(t))
        return terms.sum(axis=0)

    @pytest.mark.parametrize("resolution", [1, 2, 10, 50, 200])
    def test_matches_the_partial_fourier_sum(self, resolution):
        t = np.linspace(0, 0.5, 401)
        wave = sawtooth_v2(3.0, resolution=resolution)
        np.testing.assert_allclose(
            evaluate(wave, t), self.reference(3.0, t, resolution), atol=1e-9
        )

    def test_default_resolution_is_200_terms(self):
        t = np.linspace(0, 0.5, 101)
        np.testing.assert_allclose(
            evaluate(sawtooth_v2(3.0), t), self.reference(3.0, t, 200), atol=1e-9
        )

    def test_converges_to_a_ramp_of_slope_pi_times_frequency(self):
        # The full series equals pi*f*t on the first half period, so the
        # amplitude scale is pi/2 rather than 1 (unlike sawtooth)
        f = 2.0
        t = np.linspace(0.05, 0.45, 41) / f
        approx = evaluate(sawtooth_v2(f, resolution=200), t)
        np.testing.assert_allclose(approx, np.pi * f * t, atol=0.05)

    def test_matches_sawtooth_after_rescaling_away_from_the_jump(self):
        f = 2.0
        t = np.linspace(0.05, 0.45, 41) / f
        scaled = evaluate(sawtooth_v2(f, resolution=200), t) * 2 / np.pi
        np.testing.assert_allclose(scaled, evaluate(sawtooth(f), t), atol=0.04)

    def test_gibbs_overshoot_is_bounded(self):
        t = np.linspace(0, 1, 20_001)
        values = evaluate(sawtooth_v2(1.0, resolution=50), t)
        assert np.abs(values).max() < 1.19 * np.pi / 2

    def test_is_an_odd_function(self):
        t = np.linspace(0.01, 0.4, 25)
        wave = sawtooth_v2(2.0, resolution=30)
        np.testing.assert_allclose(evaluate(wave, -t), -evaluate(wave, t), atol=1e-9)

    def test_is_a_sum_expression_over_a_free_index(self):
        wave = sawtooth_v2(1.0, resolution=5)
        assert any(isinstance(a, sp.Sum) for a in sp.preorder_traversal(wave))


class TestTrigger:
    def test_steps_from_zero_to_one(self):
        t = np.array([0.0, 0.5, 0.999, 1.0, 1.5, 100.0])
        np.testing.assert_array_equal(evaluate(trigger(1.0), t), [0, 0, 0, 1, 1, 1])

    def test_default_trigger_time_is_one_second(self):
        assert float(trigger().subs(TIME, 0.99)) == 0.0
        assert float(trigger().subs(TIME, 1.0)) == 1.0

    @given(
        st.floats(min_value=0, max_value=10, allow_nan=False),
        st.floats(min_value=-5, max_value=15, allow_nan=False),
    )
    def test_is_the_heaviside_step(self, when, t):
        expected = 1.0 if t >= when else 0.0
        assert float(trigger(when).subs(TIME, t)) == expected

    def test_trigger_at_zero_is_always_on_for_nonnegative_time(self):
        np.testing.assert_array_equal(
            evaluate(trigger(0.0), np.array([0.0, 1.0, 5.0])), [1, 1, 1]
        )

    def test_can_gate_a_signal(self):
        gated = trigger(0.5) * sine(100)
        t = np.array([0.1, 0.3])
        np.testing.assert_allclose(evaluate(gated, t), 0)
        t = np.array([0.6])
        np.testing.assert_allclose(evaluate(gated, t), evaluate(sine(100), t))


def test_cycle_fraction_helper_is_a_modulo():
    np.testing.assert_allclose(cycle_fraction(2, [0.1, 0.6, 1.25]), [0.2, 0.2, 0.5])
