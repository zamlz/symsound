import numpy as np
import pytest

from daw.waves.basic import sine
from daw.waves.time import TIME
from daw.waves.utils import WHITE_NOISE, compile, linspace, save, white_noise


def test_linspace_excludes_endpoint():
    t = linspace(2.0, 10)
    assert len(t) == 20
    assert t[0] == 0.0
    assert t[-1] == pytest.approx(1.9)


def test_white_noise_is_bounded():
    noise = white_noise(1.0, 1000)
    assert len(noise) == 1000
    assert noise.min() >= -1.0
    assert noise.max() <= 1.0


def test_compile_matches_expression():
    sample_rate = 8000
    signal = compile(sine(100), 0.5, sample_rate)
    t = np.arange(int(0.5 * sample_rate)) / sample_rate
    assert signal == pytest.approx(np.sin(2 * np.pi * 100 * t))


def test_compile_injects_white_noise():
    signal = compile(0.5 * WHITE_NOISE + 0 * TIME, 0.1, 1000)
    assert len(signal) == 100
    assert np.abs(signal).max() <= 0.5


def test_save_writes_wav(tmp_path):
    sf = pytest.importorskip("soundfile")
    out = tmp_path / "tone.wav"
    save(sine(440), 0.1, str(out), sample_rate=8000)
    data, rate = sf.read(out)
    assert rate == 8000
    assert len(data) == 800
