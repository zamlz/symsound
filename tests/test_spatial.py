# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""pan: placing a mono signal between the ears by an angle (symsound.effects.spatial)"""

import numpy as np
import pytest
import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from helpers import ref_pan_gains
from symsound.effects import delay, pan
from symsound.waves.basic import sine
from symsound.waves.time import TIME
from symsound.waves.utils import compile

# The angle is not limited, so test far beyond a turn on either side
angles = st.floats(min_value=-1000, max_value=1000, allow_nan=False)
RATE = 8000
QUARTER = sp.pi / 4  # the angle of hard right; minus it is hard left


def gains(angle):
    pair = pan(sp.Integer(1), angle)
    return float(pair.left), float(pair.right)


class TestPan:
    def test_centre_is_one_over_root_two_each_side(self):
        pair = pan(sp.Integer(1), 0)
        assert pair.left == sp.sqrt(2) / 2
        assert pair.right == sp.sqrt(2) / 2

    def test_a_quarter_pi_is_hard_right(self):
        pair = pan(sine(440), QUARTER)
        assert pair.left == 0
        assert pair.right == sine(440)

    def test_minus_a_quarter_pi_is_hard_left(self):
        pair = pan(sine(440), -QUARTER)
        assert pair.left == sine(440)
        assert pair.right == 0

    def test_past_the_edge_one_side_turns_anti_phase(self):
        # A further quarter turn: the left is the signal upside down
        left, right = gains(sp.pi / 2)
        assert left == pytest.approx(-(2**-0.5))
        assert right == pytest.approx(2**-0.5)

    @given(angles)
    def test_gains_match_the_reference(self, angle):
        left, right = gains(angle)
        want_left, want_right = ref_pan_gains(angle)
        assert left == pytest.approx(want_left, abs=1e-9)
        assert right == pytest.approx(want_right, abs=1e-9)

    @given(angles)
    def test_power_is_the_same_at_every_angle(self, angle):
        left, right = gains(angle)
        assert left**2 + right**2 == pytest.approx(1)

    @given(angles)
    def test_a_whole_turn_changes_nothing(self, angle):
        before, after = gains(angle), gains(angle + 2 * np.pi)
        assert after == pytest.approx(before, abs=1e-9)

    @given(angles)
    def test_half_a_turn_turns_both_sides_upside_down(self, angle):
        left, right = gains(angle)
        turned_left, turned_right = gains(angle + np.pi)
        assert turned_left == pytest.approx(-left, abs=1e-9)
        assert turned_right == pytest.approx(-right, abs=1e-9)

    @given(angles)
    def test_opposite_angles_swap_the_ears(self, angle):
        left, right = gains(angle)
        mirrored_left, mirrored_right = gains(-angle)
        assert mirrored_left == pytest.approx(right, abs=1e-9)
        assert mirrored_right == pytest.approx(left, abs=1e-9)

    @given(st.floats(min_value=-np.pi / 4, max_value=np.pi / 4 - 0.01))
    def test_between_the_edges_moving_right_only_lifts_the_right_ear(self, angle):
        a, b = gains(angle), gains(angle + 0.01)
        assert b[0] <= a[0] + 1e-12
        assert b[1] >= a[1] - 1e-12

    def test_between_the_edges_neither_gain_is_negative(self):
        for angle in np.linspace(-np.pi / 4, np.pi / 4, 50):
            left, right = gains(angle)
            assert left >= -1e-12 and right >= -1e-12

    def test_degrees_can_be_written_with_rad(self):
        assert pan(sine(440), sp.rad(45)) == pan(sine(440), QUARTER)

    def test_an_exact_angle_gives_exact_gains(self):
        pair = pan(sp.Integer(1), sp.pi / 6 - sp.pi / 4)
        assert pair.left == sp.cos(sp.pi / 6)
        assert pair.right == sp.sin(sp.pi / 6)

    def test_a_complex_angle_is_rejected(self):
        with pytest.raises(ValueError, match="real number"):
            pan(sine(440), sp.I)

    def test_an_angle_with_another_symbol_is_rejected(self):
        with pytest.raises(ValueError, match="only depend on TIME"):
            pan(sine(440), sp.Symbol("p"))
        with pytest.raises(ValueError, match="only depend on TIME"):
            pan(sine(440), TIME * sp.Symbol("p"))

    def test_a_panned_signal_composes_with_effects(self):
        pair = pan(sine(440), 0.5).map(lambda s: delay(s, 0.01))
        assert pair.left.has(TIME) and pair.right.has(TIME)


class TestMovingPan:
    """The angle may be an expression of TIME, such as a slow sine"""

    def gains(self, angle, seconds=1.0):
        both = compile(pan(sp.Integer(1), angle), seconds, RATE)
        t = np.arange(len(both)) / RATE
        return t, both[:, 0], both[:, 1]

    def test_a_constant_expression_is_the_same_as_the_number(self):
        _, left, right = self.gains(0 * TIME + 0.3, 0.1)
        want_left, want_right = ref_pan_gains(0.3)
        np.testing.assert_allclose(left, want_left, atol=1e-12)
        np.testing.assert_allclose(right, want_right, atol=1e-12)

    @given(st.floats(min_value=0.5, max_value=8, allow_nan=False))
    def test_gains_follow_the_reference_at_every_sample(self, rate):
        t, left, right = self.gains(QUARTER * sine(rate), 0.5)
        angle = np.pi / 4 * np.sin(2 * np.pi * rate * t)
        want_left, want_right = ref_pan_gains(angle)
        np.testing.assert_allclose(left, want_left, atol=1e-9)
        np.testing.assert_allclose(right, want_right, atol=1e-9)

    @given(st.floats(min_value=0.5, max_value=8, allow_nan=False))
    def test_power_stays_constant_while_it_moves(self, rate):
        _, left, right = self.gains(QUARTER * sine(rate), 0.5)
        np.testing.assert_allclose(left**2 + right**2, 1, atol=1e-9)

    def test_a_slow_sine_sweeps_from_one_ear_to_the_other(self):
        _, left, right = self.gains(QUARTER * sine(1), 1.0)
        quarter, three_quarters = RATE // 4, 3 * RATE // 4
        # The angle is 0 at the start, +pi/4 a quarter of a second in (all
        # right) and -pi/4 at three quarters (all left)
        assert left[0] == pytest.approx(2**-0.5)
        assert right[quarter] == pytest.approx(1, abs=1e-3)
        assert left[quarter] == pytest.approx(0, abs=0.01)
        assert left[three_quarters] == pytest.approx(1, abs=1e-3)
        assert right[three_quarters] == pytest.approx(0, abs=0.01)

    def test_an_overshooting_angle_is_not_held_at_the_edge(self):
        # Three times the swing: it goes well past the edge, where the far
        # side changes sign, and comes back
        t, left, right = self.gains(3 * QUARTER * sine(1), 1.0)
        assert left.min() < -0.5
        angle = 3 * np.pi / 4 * np.sin(2 * np.pi * t)
        want_left, want_right = ref_pan_gains(angle)
        np.testing.assert_allclose(left, want_left, atol=1e-9)
        np.testing.assert_allclose(right, want_right, atol=1e-9)

    def test_a_steady_ramp_turns_at_a_steady_rate_without_end(self):
        # 2*pi * TIME is one whole turn a second: far outside any range, and
        # it comes back to where it started
        t, left, right = self.gains(2 * sp.pi * TIME, 3.0)
        np.testing.assert_allclose(left**2 + right**2, 1, atol=1e-9)
        for turn in (1, 2):
            at = turn * RATE
            assert left[at] == pytest.approx(left[0], abs=1e-6)
            assert right[at] == pytest.approx(right[0], abs=1e-6)
        angle = np.degrees(np.arctan2(right, left)[:2000])
        steps = np.diff(np.unwrap(np.radians(angle)))
        np.testing.assert_allclose(steps, steps[0], atol=1e-6)

    def test_a_moving_signal_is_the_signal_times_the_moving_gains(self):
        mono = compile(sine(440), 0.5, RATE)
        _, left, right = self.gains(QUARTER * sine(2), 0.5)
        both = compile(pan(sine(440), QUARTER * sine(2)), 0.5, RATE)
        np.testing.assert_allclose(both[:, 0], left * mono, atol=1e-9)
        np.testing.assert_allclose(both[:, 1], right * mono, atol=1e-9)

    def test_a_moving_pan_can_be_filtered_and_delayed_per_side(self):
        pair = pan(sine(440), QUARTER * sine(1)).map(lambda side: delay(side, 0.1))
        samples = compile(pair, 0.5, RATE)
        assert not samples[: int(0.1 * RATE)].any()
