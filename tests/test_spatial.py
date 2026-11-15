"""pan: placing a mono signal between the ears (daw.effects.spatial)"""

import numpy as np
import pytest
import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from daw.effects import delay, pan
from daw.waves.basic import sine
from daw.waves.time import TIME
from daw.waves.utils import compile
from helpers import ref_pan_gains

positions = st.floats(min_value=-1, max_value=1, allow_nan=False)
RATE = 8000


class TestPan:
    def test_hard_left_is_only_left(self):
        pair = pan(sine(440), -1)
        assert pair.left == sine(440)
        assert pair.right == 0

    def test_hard_right_is_only_right(self):
        pair = pan(sine(440), 1)
        assert pair.left == 0
        assert pair.right == sine(440)

    def test_centre_is_one_over_root_two_each_side(self):
        pair = pan(sp.Integer(1), 0)
        assert pair.left == sp.sqrt(2) / 2
        assert pair.right == sp.sqrt(2) / 2

    @given(positions)
    def test_gains_match_the_reference(self, position):
        pair = pan(sp.Integer(1), position)
        want_left, want_right = ref_pan_gains(position)
        assert float(pair.left) == pytest.approx(want_left, abs=1e-12)
        assert float(pair.right) == pytest.approx(want_right, abs=1e-12)

    @given(positions)
    def test_power_is_the_same_everywhere(self, position):
        pair = pan(sp.Integer(1), position)
        assert float(pair.left) ** 2 + float(pair.right) ** 2 == pytest.approx(1)

    @given(st.floats(min_value=-1, max_value=0.99, allow_nan=False))
    def test_moving_right_never_makes_the_left_louder(self, position):
        a = pan(sp.Integer(1), position)
        b = pan(sp.Integer(1), position + 0.01)
        assert float(b.left) <= float(a.left) + 1e-12
        assert float(b.right) >= float(a.right) - 1e-12

    def test_symmetric_positions_swap_the_ears(self):
        a, b = pan(sp.Integer(1), -0.3), pan(sp.Integer(1), 0.3)
        assert float(a.left) == pytest.approx(float(b.right))
        assert float(a.right) == pytest.approx(float(b.left))

    @pytest.mark.parametrize("position", [-1.0001, 1.0001, 2, -5])
    def test_outside_the_field_is_rejected(self, position):
        with pytest.raises(ValueError, match="between -1 and 1"):
            pan(sine(440), position)

    def test_a_complex_position_is_rejected(self):
        with pytest.raises(ValueError, match="real number"):
            pan(sine(440), sp.I)

    def test_a_position_with_another_symbol_is_rejected(self):
        with pytest.raises(ValueError, match="only depend on TIME"):
            pan(sine(440), sp.Symbol("p"))
        with pytest.raises(ValueError, match="only depend on TIME"):
            pan(sine(440), TIME * sp.Symbol("p"))

    def test_a_panned_signal_composes_with_effects(self):
        pair = pan(sine(440), 0.5).map(lambda s: delay(s, 0.01))
        assert pair.left.has(TIME) and pair.right.has(TIME)


class TestMovingPan:
    """The position may be an expression of TIME, such as a slow sine"""

    def gains(self, position, seconds=1.0):
        both = compile(pan(sp.Integer(1), position), seconds, RATE)
        t = np.arange(len(both)) / RATE
        return t, both[:, 0], both[:, 1]

    def test_a_constant_expression_is_the_same_as_the_number(self):
        assert pan(sine(440), sp.Integer(0) * TIME + sp.Rational(1, 2)) is not None
        _, left, right = self.gains(0 * TIME + 0.5, 0.1)
        want_left, want_right = ref_pan_gains(0.5)
        np.testing.assert_allclose(left, want_left, atol=1e-12)
        np.testing.assert_allclose(right, want_right, atol=1e-12)

    @given(st.floats(min_value=0.5, max_value=8, allow_nan=False))
    def test_gains_follow_the_reference_at_every_sample(self, rate):
        t, left, right = self.gains(sine(rate), 0.5)
        place = np.sin(2 * np.pi * rate * t)
        degrees = (place + 1) * 45
        np.testing.assert_allclose(left, np.cos(np.radians(degrees)), atol=1e-9)
        np.testing.assert_allclose(right, np.sin(np.radians(degrees)), atol=1e-9)

    @given(st.floats(min_value=0.5, max_value=8, allow_nan=False))
    def test_power_stays_constant_while_it_moves(self, rate):
        _, left, right = self.gains(sine(rate), 0.5)
        np.testing.assert_allclose(left**2 + right**2, 1, atol=1e-9)

    def test_a_slow_sine_sweeps_from_one_ear_to_the_other(self):
        t, left, right = self.gains(sine(1), 1.0)
        quarter, three_quarters = RATE // 4, 3 * RATE // 4
        # sine(1) is 0 at the start, +1 a quarter in (all right), -1 at three
        # quarters (all left)
        assert left[0] == pytest.approx(2**-0.5)
        assert right[quarter] == pytest.approx(1, abs=1e-3)
        assert left[quarter] == pytest.approx(0, abs=0.01)
        assert left[three_quarters] == pytest.approx(1, abs=1e-3)
        assert right[three_quarters] == pytest.approx(0, abs=0.01)

    def test_an_overshooting_position_is_held_at_the_edge(self):
        _, left, right = self.gains(3 * sine(1), 1.0)
        assert (left >= -1e-12).all() and (right >= -1e-12).all()
        np.testing.assert_allclose(left**2 + right**2, 1, atol=1e-9)
        quarter = RATE // 4
        assert right[quarter] == pytest.approx(1)
        assert left[quarter] == pytest.approx(0, abs=1e-12)

    def test_a_linear_ramp_moves_at_a_steady_rate(self):
        t, left, _ = self.gains(2 * TIME - 1, 1.0)
        degrees = np.degrees(np.arccos(left))
        np.testing.assert_allclose(np.diff(degrees), np.diff(degrees)[0], atol=1e-6)

    def test_a_moving_signal_is_the_signal_times_the_moving_gains(self):
        mono = compile(sine(440), 0.5, RATE)
        t, left, right = self.gains(sine(2), 0.5)
        both = compile(pan(sine(440), sine(2)), 0.5, RATE)
        np.testing.assert_allclose(both[:, 0], left * mono, atol=1e-9)
        np.testing.assert_allclose(both[:, 1], right * mono, atol=1e-9)

    def test_a_moving_pan_can_be_filtered_and_delayed_per_side(self):
        pair = pan(sine(440), sine(1)).map(lambda side: delay(side, 0.1))
        samples = compile(pair, 0.5, RATE)
        assert not samples[: int(0.1 * RATE)].any()
