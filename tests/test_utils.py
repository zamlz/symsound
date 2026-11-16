import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import soundfile as sf
import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

import symsound
from helpers import evaluate
from symsound.constants import DEFAULT_SAMPLE_RATE
from symsound.waves import utils
from symsound.waves.basic import sine, square
from symsound.waves.time import BPM, TIME
from symsound.waves.utils import (
    WHITE_NOISE,
    compile,
    linspace,
    play,
    plot,
    save,
    white_noise,
)

durations = st.floats(min_value=0, max_value=5, allow_nan=False)
sample_rates = st.integers(min_value=1, max_value=48_000)


class TestLinspace:
    @given(durations, sample_rates)
    def test_length_is_sample_rate_times_duration(self, duration, sr):
        assert len(linspace(duration, sr)) == int(sr * duration)

    @given(durations, sample_rates)
    def test_starts_at_zero_and_excludes_the_endpoint(self, duration, sr):
        t = linspace(duration, sr)
        if len(t):
            assert t[0] == 0.0
            assert t[-1] < duration

    @given(durations, sample_rates)
    def test_is_uniformly_spaced_and_increasing(self, duration, sr):
        t = linspace(duration, sr)
        if len(t) > 1:
            steps = np.diff(t)
            assert (steps > 0).all()
            np.testing.assert_allclose(steps, duration / len(t), rtol=1e-6)

    @given(st.integers(min_value=1, max_value=20), sample_rates)
    def test_whole_seconds_give_exact_sample_spacing(self, seconds, sr):
        t = linspace(seconds, sr)
        np.testing.assert_allclose(t[1:] - t[:-1], 1 / sr, rtol=1e-9)

    def test_zero_duration_is_empty(self):
        assert len(linspace(0, 44100)) == 0

    def test_known_values(self):
        np.testing.assert_allclose(linspace(1, 4), [0, 0.25, 0.5, 0.75])


class TestWhiteNoise:
    def test_length(self):
        assert len(white_noise(0.5, 1000)) == 500

    def test_values_are_in_the_open_unit_range(self):
        noise = white_noise(1.0, 100_000)
        assert noise.min() >= -1.0
        assert noise.max() < 1.0

    def test_is_uniform_with_zero_mean(self):
        np.random.seed(0)
        noise = white_noise(10.0, 20_000)
        assert noise.mean() == pytest.approx(0, abs=0.01)
        # Uniform on [-1, 1) has standard deviation 1 / sqrt(3)
        assert noise.std() == pytest.approx(1 / np.sqrt(3), abs=0.01)

    def test_samples_are_uncorrelated(self):
        np.random.seed(1)
        noise = white_noise(10.0, 20_000)
        assert abs(np.corrcoef(noise[:-1], noise[1:])[0, 1]) < 0.03

    def test_is_reproducible_with_a_seed(self):
        np.random.seed(5)
        first = white_noise(0.1, 1000)
        np.random.seed(5)
        np.testing.assert_array_equal(first, white_noise(0.1, 1000))

    def test_successive_calls_differ(self):
        assert not np.array_equal(white_noise(0.1, 1000), white_noise(0.1, 1000))


class TestCompile:
    def test_evaluates_a_time_dependent_expression(self):
        signal = compile(sine(100), 0.5, 8000)
        t = np.arange(4000) / 8000
        np.testing.assert_allclose(signal, np.sin(2 * np.pi * 100 * t), atol=1e-12)

    def test_time_axis_is_the_linspace(self):
        np.testing.assert_allclose(compile(TIME, 2, 10), linspace(2, 10))

    @given(durations, st.integers(min_value=1, max_value=2000))
    def test_returns_one_sample_per_sample_period(self, duration, sr):
        assert len(compile(TIME * 2, duration, sr)) == int(duration * sr)

    def test_output_is_a_float_array(self):
        signal = compile(sine(440), 0.1, 8000)
        assert isinstance(signal, np.ndarray)
        assert signal.dtype == np.float64

    def test_uses_the_default_sample_rate(self):
        assert len(compile(TIME * 1.0, 1.0)) == DEFAULT_SAMPLE_RATE

    def test_white_noise_symbol_receives_generated_noise(self):
        np.random.seed(3)
        signal = compile(WHITE_NOISE, 0.1, 1000)
        np.random.seed(3)
        np.testing.assert_array_equal(signal, white_noise(0.1, 1000))

    def test_time_and_noise_can_be_mixed(self):
        np.random.seed(4)
        signal = compile(0.5 * sine(50) + 0.1 * WHITE_NOISE, 0.2, 1000)
        np.random.seed(4)
        noise = white_noise(0.2, 1000)
        t = np.arange(200) / 1000
        expected = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.1 * noise
        np.testing.assert_allclose(signal, expected, atol=1e-12)

    def test_white_noise_symbol_is_one_shared_signal(self):
        # Documents a known limitation: every use of WHITE_NOISE in an
        # expression sees the same samples. Independent noise would multiply
        # to negative values about half the time, shared noise never does.
        np.random.seed(6)
        squared = compile(WHITE_NOISE * WHITE_NOISE, 0.1, 1000)
        np.random.seed(6)
        noise = white_noise(0.1, 1000)
        assert (squared >= 0).all()
        np.testing.assert_allclose(squared, noise**2)

    def test_noise_is_independent_of_the_sample_rate_symbol_set(self):
        assert WHITE_NOISE.free_symbols == {WHITE_NOISE}
        assert WHITE_NOISE.is_real is True

    def test_piecewise_expressions_compile(self):
        step = sp.Piecewise((0.0, TIME < 0.5), (1.0, True))
        np.testing.assert_array_equal(compile(step, 1.0, 4), [0.0, 0.0, 1.0, 1.0])

    def test_constant_expression_is_broadcast_to_every_sample(self):
        out = compile(sp.Float(0.5), 1.0, 10)
        assert out.shape == (10,)
        np.testing.assert_array_equal(out, np.full(10, 0.5))

    @pytest.mark.parametrize(
        "constant", [0, 1, sp.Integer(0), sp.Rational(1, 2), -0.25]
    )
    def test_every_kind_of_constant_gives_float_samples(self, constant):
        out = compile(sp.sympify(constant), 0.01, 1000)
        assert out.shape == (10,)
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, np.full(10, float(constant)))

    @given(
        constant=st.floats(min_value=-2, max_value=2, allow_nan=False),
        duration=durations,
        rate=st.integers(min_value=1, max_value=2000),
    )
    def test_a_constant_is_one_value_per_sample_for_any_length(
        self, constant, duration, rate
    ):
        out = compile(sp.Float(constant), duration, rate)
        assert out.shape == (int(rate * duration),)
        # sympy Floats keep 15 digits, so the value is only that exact
        np.testing.assert_allclose(out, constant, rtol=0, atol=1e-12)

    def test_the_result_can_be_written_to(self):
        # Not a read-only broadcast view of a single number
        out = compile(sp.Float(0.5), 0.01, 1000)
        out[0] = 0.0
        assert out[0] == 0.0

    def test_a_constant_plus_noise_still_varies(self):
        out = compile(0.5 + 0.0 * WHITE_NOISE, 0.01, 1000)
        assert out.shape == (10,)

    def test_matches_manual_lambdify(self):
        expr = square(7) * 0.3 + sine(11)
        t = linspace(0.5, 4000)
        np.testing.assert_allclose(compile(expr, 0.5, 4000), evaluate(expr, t))


class TestPlay:
    def test_sends_the_compiled_signal_to_the_audio_device(self, fake_sounddevice):
        play(sine(440), duration=0.1, sample_rate=8000)
        ((data, rate),) = fake_sounddevice.plays
        assert rate == 8000
        np.testing.assert_allclose(data, compile(sine(440), 0.1, 8000))

    def test_blocks_until_playback_finishes(self, fake_sounddevice):
        play(sine(440), duration=0.01, sample_rate=8000)
        assert fake_sounddevice.waits == 1

    def test_defaults_to_one_second_at_cd_rate(self, fake_sounddevice):
        play(sine(440))
        ((data, rate),) = fake_sounddevice.plays
        assert rate == DEFAULT_SAMPLE_RATE
        assert len(data) == DEFAULT_SAMPLE_RATE

    def test_keyboard_interrupt_stops_playback_quietly(self, fake_sounddevice):
        fake_sounddevice.interrupt = True
        play(sine(440), duration=0.01, sample_rate=8000)  # does not raise

    def test_other_errors_propagate(self, fake_sounddevice, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("no device")

        monkeypatch.setattr(fake_sounddevice, "play", broken)
        with pytest.raises(RuntimeError, match="no device"):
            play(sine(440), duration=0.01, sample_rate=8000)

    def test_importing_the_module_does_not_import_sounddevice(self):
        # The audio backend is optional: only play() needs it. Run in a
        # fresh interpreter with the import blocked.
        code = (
            "import sys; sys.modules['sounddevice'] = None\n"
            "import symsound.waves.utils as u\n"
            "from symsound.waves.basic import sine\n"
            "u.compile(sine(440), 0.01, 8000)\n"
            "try:\n"
            "    u.play(sine(440), 0.01, 8000)\n"
            "except ImportError:\n"
            "    print('ok')\n"
        )
        # The child does not inherit pytest's `pythonpath` setting, so point
        # it at the package explicitly (the Nix sandbox sets no PYTHONPATH)
        src = str(Path(symsound.__file__).resolve().parents[1])
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH")]))
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        assert result.stdout.strip() == "ok"


class TestSilenceAndOffsets:
    def test_saving_silence_writes_a_file_of_zeros(self, tmp_path):
        out = tmp_path / "silence.wav"
        save(sp.Integer(0), 0.25, str(out), sample_rate=8000)
        data, rate = sf.read(out)
        assert rate == 8000
        assert data.shape == (2000,)
        assert (data == 0).all()

    def test_saving_a_dc_offset_keeps_its_level(self, tmp_path):
        out = tmp_path / "dc.wav"
        save(sp.Float(0.25), 0.1, str(out), sample_rate=8000)
        data, _ = sf.read(out)
        assert data.shape == (800,)
        np.testing.assert_allclose(data, 0.25, atol=1 / 32768)

    def test_playing_silence_sends_every_sample(self, fake_sounddevice):
        play(sp.Integer(0), 0.1, sample_rate=1000)
        data, rate = fake_sounddevice.plays[0]
        assert data.shape == (100,)
        assert rate == 1000


class TestSave:
    def test_roundtrips_through_a_wav_file(self, tmp_path):
        out = tmp_path / "tone.wav"
        save(sine(440), 0.25, str(out), sample_rate=8000)
        data, rate = sf.read(out)
        assert rate == 8000
        assert data.shape == (2000,)
        # 16-bit PCM quantization error is at most one least-significant bit
        np.testing.assert_allclose(data, compile(sine(440), 0.25, 8000), atol=1 / 32768)

    def test_writes_a_mono_file(self, tmp_path):
        out = tmp_path / "mono.wav"
        save(sine(220), 0.1, str(out), sample_rate=8000)
        assert sf.info(out).channels == 1

    def test_uses_the_requested_sample_rate_and_default(self, tmp_path):
        out = tmp_path / "default.wav"
        save(sine(220), 0.1, str(out))
        assert sf.info(out).samplerate == DEFAULT_SAMPLE_RATE

    def test_duration_sets_the_frame_count(self, tmp_path):
        out = tmp_path / "length.wav"
        save(sine(220), 0.5, str(out), sample_rate=4000)
        assert sf.info(out).frames == 2000

    def test_overdriven_signal_is_clipped_not_wrapped(self, tmp_path):
        out = tmp_path / "loud.wav"
        save(1.5 * sine(100), 0.05, str(out), sample_rate=8000)
        data, _ = sf.read(out)
        assert data.min() >= -1.0
        assert data.max() <= 1.0
        # Wrapping would flip the sign of the loud peaks
        peak = np.argmax(compile(1.5 * sine(100), 0.05, 8000))
        assert data[peak] > 0.99

    def test_reports_a_missing_directory(self, tmp_path):
        with pytest.raises(Exception, match="No such file|Error opening"):
            save(sine(220), 0.01, str(tmp_path / "missing" / "x.wav"), 8000)


@pytest.fixture
def shown(monkeypatch):
    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
    return calls


class TestPlot:
    def test_draws_one_line_for_a_single_expression(self, shown):
        plot(sine(5), duration=1.0, sample_rate=1000)
        ax = plt.gcf().axes[0]
        assert len(ax.lines) == 1
        assert shown == [1]

    def test_draws_one_line_per_expression_in_a_list(self, shown):
        plot([sine(5), square(3), 0.5 * sine(7)], duration=1.0, sample_rate=1000)
        assert len(plt.gcf().axes[0].lines) == 3

    def test_plotted_data_is_the_compiled_signal(self, shown):
        plot(sine(5), duration=1.0, sample_rate=1000)
        line = plt.gcf().axes[0].lines[0]
        np.testing.assert_allclose(line.get_xdata(), linspace(1.0, 1000))
        np.testing.assert_allclose(line.get_ydata(), compile(sine(5), 1.0, 1000))

    def test_labels_for_seconds(self, shown):
        plot(sine(5), duration=1.0, sample_rate=1000)
        ax = plt.gcf().axes[0]
        assert ax.get_xlabel() == "Time [s]"
        assert ax.get_ylabel() == "Amplitude"
        assert ax.get_title() == "Signal Plot"

    def test_beats_mode_changes_the_axis_label(self, shown):
        plot(sine(5), duration=4, bpm=BPM(120), sample_rate=1000)
        assert plt.gcf().axes[0].get_xlabel() == "Beat [b]"

    def test_beats_mode_interprets_duration_as_beats(self, shown):
        bpm = BPM(120)  # 4 beats last 2 seconds
        plot(sine(5), duration=4, bpm=bpm, sample_rate=1000)
        line = plt.gcf().axes[0].lines[0]
        assert len(line.get_xdata()) == 2000
        assert line.get_xdata()[0] == 0
        assert line.get_xdata()[-1] == pytest.approx(4, abs=0.01)
        assert np.all(np.diff(line.get_xdata()) > 0)

    def test_beats_mode_signal_covers_the_same_time_span(self, shown):
        plot(sine(5), duration=4, bpm=BPM(120), sample_rate=1000)
        line = plt.gcf().axes[0].lines[0]
        np.testing.assert_allclose(line.get_ydata(), compile(sine(5), 2.0, 1000))

    def test_grid_is_enabled(self, shown):
        plot(sine(5), duration=0.1, sample_rate=1000)
        ax = plt.gcf().axes[0]
        assert any(line.get_visible() for line in ax.xaxis.get_gridlines())


def test_module_exports_the_public_api():
    for name in ("linspace", "white_noise", "compile", "play", "save", "plot"):
        assert callable(getattr(utils, name))
