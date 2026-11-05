import pytest

from daw.envelope import ACTIVE_DURATION, adsr_envelope
from daw.waves.time import TIME


@pytest.fixture
def envelope():
    env = adsr_envelope(attack=0.1, decay=0.1, sustain=0.5, release=0.2)
    return env.subs(ACTIVE_DURATION, 1.0)


@pytest.mark.parametrize(
    ("t", "expected"),
    [
        (0.05, 0.5),  # halfway through attack
        (0.1, 1.0),  # attack peak
        (0.2, 0.5),  # end of decay
        (0.5, 0.5),  # sustain
        (1.1, 0.25),  # halfway through release
        (2.0, 0.0),  # after release
    ],
)
def test_adsr_shape(envelope, t, expected):
    assert float(envelope.subs(TIME, t)) == pytest.approx(expected)


@pytest.mark.parametrize(
    "args", [(-1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, -1)]
)
def test_negative_parameters_are_rejected(args):
    with pytest.raises(Exception, match="less than zero"):
        adsr_envelope(*args)


def test_zero_attack_starts_at_full_level():
    env = adsr_envelope(0, 0.1, 0.5, 0.1).subs(ACTIVE_DURATION, 1.0)
    assert float(env.subs(TIME, 0.0)) == pytest.approx(1.0)
