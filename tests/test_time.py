import pytest

from daw.waves.time import BPM


def test_time_of_one_beat_at_120_bpm():
    assert BPM(120).time(1) == pytest.approx(0.5)


def test_frequency_is_inverse_of_time():
    bpm = BPM(140)
    assert bpm.frequency(4) == pytest.approx(1 / bpm.time(4))


def test_beat_inverts_time():
    bpm = BPM(95)
    assert bpm.beat(bpm.time(3.5)) == pytest.approx(3.5)
