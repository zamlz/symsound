import math

import pytest
import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

from symsound.waves.time import BPM, TIME

bpms = st.floats(min_value=1, max_value=400, allow_nan=False)
beats = st.floats(min_value=-64, max_value=64, allow_nan=False)
positive_beats = st.floats(min_value=1e-3, max_value=64, allow_nan=False)


class TestTimeSymbol:
    def test_is_real(self):
        assert TIME.is_real is True

    def test_is_not_declared_positive(self):
        # A positive TIME makes sympy evaluate `TIME < 0` to False when an
        # expression is built, silently dropping guards for negative time.
        # Shifting a note with TIME -> TIME - start then breaks. Regression
        # test for the noisy ode_to_joy bug.
        assert TIME.is_positive is None
        assert TIME.is_nonnegative is None

    def test_negative_time_guards_are_preserved(self):
        assert isinstance(TIME < 0, sp.core.relational.Relational)
        piecewise = sp.Piecewise((0.0, TIME < 0), (1.0, True))
        assert len(piecewise.args) == 2

    def test_symbol_name(self):
        assert TIME.name == "time"


class TestBPM:
    @pytest.mark.parametrize(
        ("bpm", "beats", "seconds"),
        [(60, 1, 1.0), (120, 1, 0.5), (120, 4, 2.0), (140, 0, 0.0), (90, 3, 2.0)],
    )
    def test_time_known_values(self, bpm, beats, seconds):
        assert BPM(bpm).time(beats) == pytest.approx(seconds)

    @pytest.mark.parametrize(
        ("bpm", "beats", "hertz"),
        [(60, 1, 1.0), (120, 1, 2.0), (120, 4, 0.5), (240, 2, 2.0)],
    )
    def test_frequency_known_values(self, bpm, beats, hertz):
        assert BPM(bpm).frequency(beats) == pytest.approx(hertz)

    @pytest.mark.parametrize(
        ("bpm", "seconds", "beats"),
        [(60, 1, 1.0), (120, 1, 2.0), (120, 0.25, 0.5), (150, 2, 5.0)],
    )
    def test_beat_known_values(self, bpm, seconds, beats):
        assert BPM(bpm).beat(seconds) == pytest.approx(beats)

    def test_accepts_int_and_float(self):
        assert BPM(120).time(1) == BPM(120.0).time(1.0)

    @given(bpms, beats)
    def test_beat_inverts_time(self, bpm, n):
        b = BPM(bpm)
        assert b.beat(b.time(n)) == pytest.approx(n, abs=1e-9)

    @given(bpms, beats)
    def test_time_inverts_beat(self, bpm, seconds):
        b = BPM(bpm)
        assert b.time(b.beat(seconds)) == pytest.approx(seconds, abs=1e-9)

    @given(bpms, positive_beats)
    def test_frequency_is_reciprocal_of_time(self, bpm, n):
        b = BPM(bpm)
        assert b.frequency(n) * b.time(n) == pytest.approx(1.0)

    @given(bpms, beats, beats)
    def test_time_is_linear(self, bpm, a, b):
        t = BPM(bpm).time
        assert t(a + b) == pytest.approx(t(a) + t(b), abs=1e-9)

    @given(bpms, positive_beats)
    def test_doubling_tempo_halves_duration(self, bpm, n):
        assert BPM(2 * bpm).time(n) == pytest.approx(BPM(bpm).time(n) / 2)

    @given(bpms)
    def test_one_beat_lasts_sixty_over_bpm_seconds(self, bpm):
        assert BPM(bpm).time(1) == pytest.approx(60 / bpm)

    def test_one_beat_at_60_bpm_has_a_period_of_one_second(self):
        assert math.isclose(BPM(60).frequency(1), 1.0)
