import numpy as np
import pytest
import sympy as sp
from hypothesis import assume, given
from hypothesis import strategies as st

from daw.envelope import ACTIVE_DURATION, adsr_envelope
from daw.waves.time import TIME
from helpers import ref_adsr

# Durations are either exactly zero (a degenerate stage) or clearly positive
stage = st.one_of(st.just(0.0), st.floats(min_value=0.001, max_value=2))
positive_stage = st.floats(min_value=0.001, max_value=2)
levels = st.floats(min_value=0, max_value=1)
holds = st.floats(min_value=0, max_value=5)


def envelope_function(a, d, s, r):
    env = adsr_envelope(a, d, s, r)
    fn = sp.lambdify((TIME, ACTIVE_DURATION), env, "numpy")

    def call(t, active):
        t = np.asarray(t, dtype=float)
        return np.broadcast_to(np.asarray(fn(t, active), dtype=float), t.shape)

    return call


def away_from_stage_edges(a, d, active):
    """sympy stores Floats with 15 significant digits, so a hold that lands
    within ~1e-12 of a stage edge can fall on either side of it. That only
    matters where the level jumps (e.g. decay 0, sustain < 1), which makes the
    release start level ill-defined there. Not a behaviour worth pinning."""
    assume(abs(active - a) > 1e-9 and abs(active - (a + d)) > 1e-9)


def sample_times(a, d, r, active):
    end = active + r + 1.0
    base = np.linspace(-0.5, end, 401)
    edges = [a, a + d, active, active + r]
    return np.sort(np.concatenate([base, np.array(edges) + [1e-6, -1e-6, 1e-6, -1e-6]]))


@pytest.fixture
def envelope():
    env = adsr_envelope(attack=0.1, decay=0.1, sustain=0.5, release=0.2)
    return env.subs(ACTIVE_DURATION, 1.0)


class TestKnownShape:
    @pytest.mark.parametrize(
        ("t", "expected"),
        [
            (-1.0, 0.0),  # before the note
            (0.0, 0.0),  # start
            (0.05, 0.5),  # halfway through attack
            (0.1, 1.0),  # attack peak
            (0.15, 0.75),  # halfway through decay
            (0.2, 0.5),  # end of decay
            (0.5, 0.5),  # sustain
            (1.0, 0.5),  # last instant of the note
            (1.1, 0.25),  # halfway through release
            (1.2, 0.0),  # end of release
            (2.0, 0.0),  # after release
        ],
    )
    def test_adsr_shape(self, envelope, t, expected):
        assert float(envelope.subs(TIME, t)) == pytest.approx(expected, abs=1e-12)

    def test_free_symbols_are_time_and_active_duration(self):
        assert adsr_envelope(0.1, 0.1, 0.5, 0.1).free_symbols == {
            TIME,
            ACTIVE_DURATION,
        }


class TestDegenerateStages:
    def test_zero_attack_starts_at_full_level(self):
        env = adsr_envelope(0, 0.1, 0.5, 0.1).subs(ACTIVE_DURATION, 1.0)
        assert float(env.subs(TIME, 0.0)) == pytest.approx(1.0)
        assert float(env.subs(TIME, 0.05)) == pytest.approx(0.75)

    def test_zero_decay_jumps_straight_to_sustain(self):
        env = adsr_envelope(0.1, 0, 0.5, 0.1).subs(ACTIVE_DURATION, 1.0)
        assert float(env.subs(TIME, 0.1)) == pytest.approx(1.0)
        assert float(env.subs(TIME, 0.1 + 1e-9)) == pytest.approx(0.5)

    def test_zero_attack_and_decay_is_a_pure_sustain(self):
        env = adsr_envelope(0, 0, 0.5, 0.1).subs(ACTIVE_DURATION, 1.0)
        assert [float(env.subs(TIME, t)) for t in (0.0, 0.01, 0.5, 1.0)] == [
            1.0,
            0.5,
            0.5,
            0.5,
        ]

    def test_zero_release_cuts_off_at_the_end_of_the_note(self):
        env = adsr_envelope(0.1, 0.1, 0.5, 0).subs(ACTIVE_DURATION, 1.0)
        assert float(env.subs(TIME, 1.0)) == pytest.approx(0.5)
        assert float(env.subs(TIME, 1.0 + 1e-9)) == 0.0

    def test_all_stages_zero_is_a_gate(self):
        env = adsr_envelope(0, 0, 1.0, 0).subs(ACTIVE_DURATION, 1.0)
        assert float(env.subs(TIME, 0.5)) == 1.0
        assert float(env.subs(TIME, 1.5)) == 0.0

    def test_full_sustain_has_no_audible_decay(self):
        env = adsr_envelope(0.1, 0.1, 1.0, 0.1).subs(ACTIVE_DURATION, 1.0)
        assert [float(env.subs(TIME, t)) for t in (0.1, 0.15, 0.5)] == [1.0] * 3

    def test_zero_sustain_decays_to_silence_while_held(self):
        env = adsr_envelope(0.1, 0.1, 0.0, 0.1).subs(ACTIVE_DURATION, 1.0)
        assert float(env.subs(TIME, 0.2)) == pytest.approx(0.0)
        assert float(env.subs(TIME, 0.5)) == 0.0


class TestReleaseFromAnyStage:
    """The release starts from whatever level the envelope reached"""

    def test_released_during_attack(self):
        env = adsr_envelope(0.1, 0.1, 0.5, 0.1).subs(ACTIVE_DURATION, 0.05)
        assert float(env.subs(TIME, 0.05)) == pytest.approx(0.5)
        assert float(env.subs(TIME, 0.1)) == pytest.approx(0.25)
        assert float(env.subs(TIME, 0.15)) == pytest.approx(0.0, abs=1e-12)

    def test_released_during_decay(self):
        env = adsr_envelope(0.1, 0.1, 0.5, 0.1).subs(ACTIVE_DURATION, 0.15)
        assert float(env.subs(TIME, 0.15)) == pytest.approx(0.75)
        assert float(env.subs(TIME, 0.2)) == pytest.approx(0.375)
        assert float(env.subs(TIME, 0.25)) == pytest.approx(0.0, abs=1e-12)

    def test_zero_length_note_is_silent_after_zero(self):
        env = adsr_envelope(0.1, 0.1, 0.5, 0.1).subs(ACTIVE_DURATION, 0.0)
        assert float(env.subs(TIME, 0.0)) == 0.0
        assert float(env.subs(TIME, 0.5)) == 0.0


class TestAgainstReferenceImplementation:
    @given(stage, stage, levels, stage, holds, st.data())
    def test_matches_independent_definition(self, a, d, s, r, active, data):
        away_from_stage_edges(a, d, active)
        t = np.array(
            data.draw(
                st.lists(
                    st.floats(min_value=-1, max_value=active + r + 2),
                    min_size=1,
                    max_size=20,
                )
            )
        )
        edges = np.array([0.0, a, a + d, active, active + r])
        assume(np.abs(t[:, None] - edges[None, :]).min() > 1e-9)
        np.testing.assert_allclose(
            envelope_function(a, d, s, r)(t, active),
            ref_adsr(a, d, s, r, active, t),
            atol=1e-9,
        )

    @given(stage, stage, levels, stage, holds)
    def test_dense_grid_matches_reference(self, a, d, s, r, active):
        away_from_stage_edges(a, d, active)
        t = sample_times(a, d, r, active)
        np.testing.assert_allclose(
            envelope_function(a, d, s, r)(t, active),
            ref_adsr(a, d, s, r, active, t),
            atol=1e-9,
        )


class TestInvariants:
    @given(stage, stage, levels, stage, holds)
    def test_stays_within_zero_and_one(self, a, d, s, r, active):
        values = envelope_function(a, d, s, r)(sample_times(a, d, r, active), active)
        assert np.isfinite(values).all()
        assert values.min() >= -1e-9
        assert values.max() <= 1 + 1e-9

    @given(stage, stage, levels, stage, holds)
    def test_silent_before_the_note(self, a, d, s, r, active):
        t = np.linspace(-5, -1e-9, 50)
        assert not envelope_function(a, d, s, r)(t, active).any()

    @given(stage, stage, levels, stage, holds)
    def test_silent_after_the_release(self, a, d, s, r, active):
        t = active + r + np.linspace(1e-6, 5, 50)
        assert not envelope_function(a, d, s, r)(t, active).any()

    @given(positive_stage, stage, levels, stage, holds)
    def test_starts_from_zero_when_there_is_an_attack(self, a, d, s, r, active):
        assert envelope_function(a, d, s, r)(np.array([0.0]), active)[0] == 0.0

    @given(positive_stage, positive_stage, levels, positive_stage, holds)
    def test_is_continuous_with_all_stages_present(self, a, d, s, r, active):
        t = sample_times(a, d, r, active)
        values = envelope_function(a, d, s, r)(t, active)
        # Largest possible slope is 1/min(a, d, r); allow a little slack
        max_slope = 1 / min(a, d, r)
        steps = np.abs(np.diff(values))
        assert (steps <= max_slope * np.diff(t) + 1e-9).all()

    @given(positive_stage, positive_stage, levels, stage, holds)
    def test_attack_rises_then_decay_falls(self, a, d, s, r, active):
        assume(active > a + d)
        f = envelope_function(a, d, s, r)
        attack = f(np.linspace(0, a, 25), active)
        decay = f(np.linspace(a, a + d, 25), active)
        assert np.all(np.diff(attack) >= -1e-12)
        assert np.all(np.diff(decay) <= 1e-12)
        assert attack[-1] == pytest.approx(1.0, abs=1e-9)
        assert decay[-1] == pytest.approx(s, abs=1e-9)

    @given(stage, stage, levels, positive_stage, holds)
    def test_release_falls_monotonically_to_zero(self, a, d, s, r, active):
        f = envelope_function(a, d, s, r)
        release = f(active + np.linspace(1e-9, r, 25), active)
        assert np.all(np.diff(release) <= 1e-12)
        assert release[-1] == pytest.approx(0.0, abs=1e-9)

    @given(stage, stage, levels, stage, holds)
    def test_holds_sustain_level_between_decay_and_release(self, a, d, s, r, active):
        assume(active > a + d + 1e-3)
        t = np.linspace(a + d + 1e-6, active, 25)
        np.testing.assert_allclose(envelope_function(a, d, s, r)(t, active), s)


class TestTimeShifting:
    """Notes are placed in a song by substituting TIME -> TIME - start"""

    @given(
        stage,
        stage,
        levels,
        stage,
        st.floats(min_value=0.0, max_value=3),
        st.floats(min_value=0.0, max_value=10),
    )
    def test_shifted_envelope_matches_the_unshifted_one(self, a, d, s, r, active, t0):
        env = adsr_envelope(a, d, s, r).subs(ACTIVE_DURATION, active)
        shifted = sp.lambdify(TIME, env.subs(TIME, TIME - t0), "numpy")
        plain = sp.lambdify(TIME, env, "numpy")
        t = np.linspace(-1, t0 + active + r + 1, 301)
        # sympy may rewrite `TIME - t0 <= x` as `TIME <= x + t0`, which rounds
        # differently exactly on a stage boundary, so skip those instants
        edges = np.array([0.0, a, a + d, active, active + r]) + t0
        t = t[np.abs(t[:, None] - edges[None, :]).min(axis=1) > 1e-7]
        np.testing.assert_allclose(
            np.broadcast_to(shifted(t), t.shape),
            np.broadcast_to(plain(t - t0), t.shape),
            atol=1e-9,
        )

    @given(stage, stage, levels, stage, holds, st.floats(min_value=0.01, max_value=10))
    def test_shifted_envelope_is_silent_before_it_starts(self, a, d, s, r, active, t0):
        env = adsr_envelope(a, d, s, r).subs(ACTIVE_DURATION, active)
        shifted = sp.lambdify(TIME, env.subs(TIME, TIME - t0), "numpy")
        t = np.linspace(-5, t0 - 1e-6, 100)
        assert not np.broadcast_to(shifted(t), t.shape).any()

    def test_negative_time_branch_survives_construction(self):
        # Regression: a positive TIME symbol made sympy discard the
        # `TIME < 0 -> 0` branch, so shifted notes ramped up before starting
        env = adsr_envelope(0.1, 0.1, 0.5, 0.1)
        inner = env.args[0][0]
        conditions = [str(c) for _, c in inner.args]
        assert any("time < 0" in c for c in conditions), conditions

    def test_ode_to_joy_style_stacking_stays_bounded(self):
        env = adsr_envelope(0.1, 0.1, 0.5, 0.1)
        total = 0
        for start in np.arange(0, 6, 0.375):
            total += env.subs(ACTIVE_DURATION, 0.1875).subs(TIME, TIME - start)
        t = np.linspace(0, 8, 4000)
        values = np.broadcast_to(sp.lambdify(TIME, total, "numpy")(t), t.shape)
        assert values.min() >= -1e-9
        assert values.max() <= 2.0


class TestValidation:
    @pytest.mark.parametrize(
        "args", [(-1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, -1)]
    )
    def test_negative_parameters_are_rejected(self, args):
        with pytest.raises(Exception, match="less than zero"):
            adsr_envelope(*args)

    @given(
        st.floats(min_value=-10, max_value=-1e-9),
        stage,
        levels,
        stage,
    )
    def test_any_negative_attack_is_rejected(self, bad, d, s, r):
        with pytest.raises(Exception, match="less than zero"):
            adsr_envelope(bad, d, s, r)

    @pytest.mark.parametrize("position", range(4))
    def test_zero_is_a_valid_value_for_every_parameter(self, position):
        args = [0.1, 0.1, 0.5, 0.1]
        args[position] = 0.0
        adsr_envelope(*args)
