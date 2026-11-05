import pytest

from daw.waves.basic import sawtooth, sine, square, triangle, trigger
from daw.waves.time import TIME


def value(expr, t):
    return float(expr.subs(TIME, t))


def test_sine_quarter_period():
    assert value(sine(1), 0.25) == pytest.approx(1.0)
    assert value(sine(1), 0.75) == pytest.approx(-1.0)


def test_sine_frequency_scales_period():
    assert value(sine(440), 1 / 440 / 4) == pytest.approx(1.0)


def test_square_alternates_sign():
    assert value(square(1), 0.25) == 1.0
    assert value(square(1), 0.75) == -1.0


def test_triangle_extremes():
    assert value(triangle(1), 0.0) == pytest.approx(1.0)
    assert value(triangle(1), 0.5) == pytest.approx(-1.0)


def test_sawtooth_ramps_each_period():
    assert value(sawtooth(1), 0.0) == pytest.approx(0.0)
    assert value(sawtooth(1), 0.25) == pytest.approx(0.5)
    assert value(sawtooth(1), 1.25) == pytest.approx(0.5)


def test_time_shift_advances_signal():
    shifted = sine(1, time_shift=TIME + 0.25)
    assert value(shifted, 0.0) == pytest.approx(1.0)


def test_trigger_steps_at_trigger_time():
    t = trigger(2.0)
    assert value(t, 1.9) == 0.0
    assert value(t, 2.0) == 1.0
