import numpy as np
import pytest
import soundfile as sf
import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from daw.effects import delay
from daw.stereo import StereoExpr, mono_to_stereo, pan
from daw.waves.basic import sine
from daw.waves.time import TIME
from daw.waves.utils import WHITE_NOISE, compile, play, save
from helpers import evaluate

positions = st.floats(min_value=-1, max_value=1, allow_nan=False)
RATE = 8000


def ref_pan_gains(position):
    """Equal-power gains written with a plain angle in degrees: 0 is all left,
    90 all right"""
    degrees = (position + 1) * 45
    return np.cos(np.radians(degrees)), np.sin(np.radians(degrees))


class TestStereo:
    def test_holds_two_expressions(self):
        pair = StereoExpr(sine(440), sine(660))
        assert pair.left == sine(440)
        assert pair.right == sine(660)

    def test_numbers_are_accepted(self):
        pair = StereoExpr(1, 0.5)
        assert pair.left == sp.Integer(1)
        assert pair.right == sp.Float(0.5)

    def test_is_immutable(self):
        with pytest.raises(AttributeError):
            StereoExpr(1, 2).left = 3  # type: ignore[misc]

    def test_map_applies_the_function_to_both_sides(self):
        pair = StereoExpr(sine(440), sine(660)).map(lambda s: 2 * s)
        assert pair == StereoExpr(2 * sine(440), 2 * sine(660))

    def test_mapping_delay_delays_each_side(self):
        pair = StereoExpr(sine(440), sine(660)).map(lambda s: delay(s, 0.01))
        assert pair.left == delay(sine(440), 0.01)
        assert pair.right == delay(sine(660), 0.01)

    def test_stereo_puts_a_mono_signal_in_both_ears(self):
        pair = mono_to_stereo(sine(440))
        assert pair.left == pair.right == sine(440)


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


class TestCompileStereo:
    def test_shape_is_samples_by_two(self):
        samples = compile(StereoExpr(sine(440), sine(660)), 0.5, RATE)
        assert samples.shape == (RATE // 2, 2)
        assert samples.dtype == np.float64

    def test_each_column_is_that_side_compiled_alone(self):
        pair = StereoExpr(sine(440), sine(660))
        both = compile(pair, 0.25, RATE)
        np.testing.assert_array_equal(both[:, 0], compile(pair.left, 0.25, RATE))
        np.testing.assert_array_equal(both[:, 1], compile(pair.right, 0.25, RATE))

    def test_columns_match_the_formula(self):
        t = np.arange(RATE // 4) / RATE
        samples = compile(StereoExpr(sine(440), sine(660)), 0.25, RATE)
        np.testing.assert_allclose(
            samples[:, 0], np.sin(2 * np.pi * 440 * t), atol=1e-9
        )
        np.testing.assert_allclose(
            samples[:, 1], np.sin(2 * np.pi * 660 * t), atol=1e-9
        )

    def test_a_constant_side_has_one_value_per_sample(self):
        samples = compile(StereoExpr(sp.Integer(0), sp.Float(0.25)), 0.1, RATE)
        assert samples.shape == (RATE // 10, 2)
        assert (samples[:, 0] == 0).all()
        assert (samples[:, 1] == 0.25).all()

    def test_zero_duration_is_empty(self):
        assert compile(StereoExpr(sine(440), sine(440)), 0, RATE).shape == (0, 2)

    def test_both_sides_share_the_same_noise(self):
        samples = compile(mono_to_stereo(WHITE_NOISE), 0.1, RATE)
        assert samples[:, 0].std() > 0.1
        np.testing.assert_array_equal(samples[:, 0], samples[:, 1])

    def test_noise_is_not_repeated_over_time(self):
        samples = compile(mono_to_stereo(WHITE_NOISE), 0.1, RATE)
        assert len(np.unique(samples[:, 0])) > len(samples) // 2

    def test_a_mono_signal_still_compiles_to_one_dimension(self):
        samples = compile(sine(440), 0.25, RATE)
        assert samples.shape == (RATE // 4,)

    def test_a_mono_signal_is_unchanged_by_stereo_support(self):
        samples = compile(sine(440) * 0.5, 0.25, RATE)
        want = evaluate(sine(440) * 0.5, np.arange(RATE // 4) / RATE)
        np.testing.assert_allclose(samples, want, atol=1e-12)

    @given(positions)
    def test_stereo_of_a_signal_is_the_signal_in_both_columns(self, position):
        mono = compile(sine(440), 0.05, RATE)
        both = compile(mono_to_stereo(sine(440)), 0.05, RATE)
        np.testing.assert_array_equal(both, np.stack([mono, mono], axis=1))

    @given(positions)
    def test_panned_columns_are_the_gains_times_the_signal(self, position):
        mono = compile(sine(440), 0.05, RATE)
        both = compile(pan(sine(440), position), 0.05, RATE)
        gain_left, gain_right = ref_pan_gains(position)
        np.testing.assert_allclose(both[:, 0], gain_left * mono, atol=1e-9)
        np.testing.assert_allclose(both[:, 1], gain_right * mono, atol=1e-9)

    def test_a_hard_panned_tone_is_silent_in_the_other_ear(self):
        left = compile(pan(sine(440), -1), 0.25, RATE)
        right = compile(pan(sine(440), 1), 0.25, RATE)
        assert np.abs(left[:, 0]).max() == pytest.approx(1, abs=0.01)
        assert not left[:, 1].any()
        assert not right[:, 0].any()
        assert np.abs(right[:, 1]).max() == pytest.approx(1, abs=0.01)

    def test_a_centred_tone_is_3_db_down_in_each_ear(self):
        both = compile(pan(sine(440), 0), 0.25, RATE)
        assert np.abs(both).max() == pytest.approx(2**-0.5, abs=0.01)
        rms = np.sqrt((both**2).mean(axis=0))
        np.testing.assert_allclose(rms, 0.5, atol=0.01)

    def test_a_delayed_side_is_silent_until_its_delay(self):
        pair = StereoExpr(sine(440), delay(sine(440), 0.1))
        samples = compile(pair, 0.25, RATE)
        first = int(0.1 * RATE)
        assert np.abs(samples[:, 0][:first]).max() > 0.5
        assert not samples[:first, 1].any()
        np.testing.assert_allclose(
            samples[first:, 1], samples[: len(samples) - first, 0], atol=1e-9
        )

    def test_a_filter_on_one_side_starts_at_rest(self):
        from daw.filters import low_pass_filter

        pair = StereoExpr(sine(100), low_pass_filter(sine(100), 50))
        samples = compile(pair, 0.25, RATE)
        assert samples[0, 1] == 0
        assert np.abs(samples[:, 1]).max() < np.abs(samples[:, 0]).max()


class TestOutput:
    def test_save_writes_two_channels(self, tmp_path):
        out = tmp_path / "pair.wav"
        save(StereoExpr(sine(440), sine(660)), 0.25, str(out), sample_rate=RATE)
        info = sf.info(out)
        assert info.channels == 2
        assert info.frames == RATE // 4
        assert info.samplerate == RATE

    def test_saved_channels_round_trip(self, tmp_path):
        out = tmp_path / "pair.wav"
        pair = pan(sine(440), -0.5)
        save(pair, 0.25, str(out), sample_rate=RATE)
        data, _ = sf.read(out, dtype="float64")
        want = compile(pair, 0.25, RATE)
        assert data.shape == want.shape
        # WAV is 16 bit: one step is 1 / 32768
        np.testing.assert_allclose(data, want, atol=2 / 32768)

    def test_a_hard_panned_file_is_silent_on_one_channel(self, tmp_path):
        out = tmp_path / "left.wav"
        save(pan(sine(440), -1), 0.25, str(out), sample_rate=RATE)
        data, _ = sf.read(out)
        assert not data[:, 1].any()
        assert np.abs(data[:, 0]).max() > 0.9

    def test_a_mono_file_is_still_mono(self, tmp_path):
        out = tmp_path / "mono.wav"
        save(sine(440), 0.25, str(out), sample_rate=RATE)
        assert sf.info(out).channels == 1

    def test_play_sends_both_channels(self, fake_sounddevice):
        play(StereoExpr(sine(440), sine(660)), 0.1, sample_rate=RATE)
        ((data, rate),) = fake_sounddevice.plays
        assert data.shape == (RATE // 10, 2)
        assert rate == RATE
        assert fake_sounddevice.waits == 1
