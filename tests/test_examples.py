"""Run every example script end to end with audio stubbed out

Examples are the closest thing this project has to acceptance tests: they
exercise the whole pipeline (symbolic construction -> compile -> play/save).
"""

import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from daw.constants import DEFAULT_SAMPLE_RATE, Note
from daw.waves.utils import compile
from golden import check_golden
from helpers import evaluate

pytestmark = pytest.mark.slow

EXAMPLES = Path(__file__).parent.parent / "examples"
SCRIPTS = sorted(p.stem for p in EXAMPLES.glob("*.py"))


def run_example(name, fake_sounddevice, tmp_path, monkeypatch):
    """Run an example in a scratch directory and return its namespace"""
    monkeypatch.chdir(tmp_path)
    np.random.seed(0)
    return runpy.run_path(str(EXAMPLES / f"{name}.py"), run_name="__main__")


def test_every_example_is_covered_by_this_file():
    assert set(SCRIPTS) == {
        "c_major_shimmer",
        "classic_synths",
        "house",
        "house_with_low_pass",
        "ode_to_joy",
        "weighted_blend_test",
    }, "add new examples to the tests below"


@pytest.mark.parametrize("name", SCRIPTS)
def test_example_runs_without_error(name, fake_sounddevice, tmp_path, monkeypatch):
    run_example(name, fake_sounddevice, tmp_path, monkeypatch)


@pytest.mark.parametrize("name", ["house", "house_with_low_pass"])
def test_house_does_not_clip(name, fake_sounddevice, tmp_path, monkeypatch):
    run_example(name, fake_sounddevice, tmp_path, monkeypatch)
    for data, _ in fake_sounddevice.plays:
        # Samples beyond 1 are clipped on playback and in the WAV file
        assert np.abs(data).max() <= 1.0
        # ...and the 0.8 gain should not have turned it down needlessly
        assert np.abs(data).max() > 0.9


@pytest.mark.parametrize("name", [n for n in SCRIPTS if n != "classic_synths"])
def test_example_plays_finite_audible_audio(
    name, fake_sounddevice, tmp_path, monkeypatch
):
    run_example(name, fake_sounddevice, tmp_path, monkeypatch)
    assert fake_sounddevice.plays, "example never played anything"
    for data, rate in fake_sounddevice.plays:
        assert rate == DEFAULT_SAMPLE_RATE
        assert data.ndim == 1
        assert np.isfinite(data).all()
        assert np.sqrt((data**2).mean()) > 1e-3, "example is silent"
        assert np.abs(data).max() < 2.0, "example is far too loud"


def test_classic_synths_builds_a_filtered_signal_without_playing(
    fake_sounddevice, tmp_path, monkeypatch
):
    ns = run_example("classic_synths", fake_sounddevice, tmp_path, monkeypatch)
    assert fake_sounddevice.plays == []
    assert ns["w2"].free_symbols


def test_classic_synths_filter_attenuates_440_hz_by_the_expected_amount(
    fake_sounddevice, tmp_path, monkeypatch
):
    ns = run_example("classic_synths", fake_sounddevice, tmp_path, monkeypatch)
    out = compile(ns["w2"], 1.0)
    assert np.isfinite(out).all()
    # 440 Hz through a cutoff of about 150 Hz: gain 1/sqrt(1 + (440/150)^2)
    # = 0.32, so the RMS of a unit sine (0.707) drops to about 0.23
    rms = np.sqrt((out[DEFAULT_SAMPLE_RATE // 4 :] ** 2).mean())
    assert rms == pytest.approx(0.23, rel=0.1)
    assert np.abs(out).max() < 0.5


@pytest.mark.parametrize("name", ["house", "house_with_low_pass"])
def test_house_writes_its_wav_file(name, fake_sounddevice, tmp_path, monkeypatch):
    run_example(name, fake_sounddevice, tmp_path, monkeypatch)
    info = sf.info(tmp_path / "beat.wav")
    assert info.samplerate == DEFAULT_SAMPLE_RATE
    assert info.frames == 30 * DEFAULT_SAMPLE_RATE
    assert info.channels == 1


@pytest.fixture(scope="module")
def house(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        mp.setitem(sys.modules, "sounddevice", _SilentSoundDevice)
        mp.chdir(tmp_path_factory.mktemp("house"))
        np.random.seed(0)
        ns = runpy.run_path(
            str(EXAMPLES / "house_with_low_pass.py"), run_name="__main__"
        )
    finally:
        mp.undo()
    t = np.arange(30 * DEFAULT_SAMPLE_RATE) / DEFAULT_SAMPLE_RATE
    noise = np.random.default_rng(0).uniform(-1, 1, len(t))
    return {
        "ns": ns,
        "t": t,
        "dry": evaluate(ns["dry"], t, noise),
        "wet": evaluate(ns["w"], t, noise),
    }


class TestHouseFilterSweep:
    """house_with_low_pass.py: 30 s, dry for 10 s, a low-pass sweep, dry again"""

    SR = DEFAULT_SAMPLE_RATE

    def band_power(self, signal, start, stop, low_hz):
        """Power above low_hz in the seconds [start, stop)"""
        chunk = signal[int(start * self.SR) : int(stop * self.SR)]
        spectrum = np.abs(np.fft.rfft(chunk)) ** 2
        freqs = np.fft.rfftfreq(len(chunk), 1 / self.SR)
        return spectrum[freqs > low_hz].sum()

    def treble_ratio(self, house, start, stop):
        return self.band_power(house["wet"], start, stop, 2000) / self.band_power(
            house["dry"], start, stop, 2000
        )

    def test_is_thirty_seconds_long(self, house):
        assert len(house["wet"]) == 30 * self.SR

    def test_first_and_last_thirds_are_exactly_unfiltered(self, house):
        diff = np.abs(house["wet"] - house["dry"])
        assert diff[: 10 * self.SR].max() == 0.0
        assert diff[20 * self.SR :].max() == pytest.approx(0.0, abs=1e-12)

    def test_the_middle_third_is_filtered(self, house):
        diff = np.abs(house["wet"] - house["dry"])[10 * self.SR : 20 * self.SR]
        assert diff.max() > 0.05

    def test_treble_falls_to_almost_nothing_at_the_centre(self, house):
        assert self.treble_ratio(house, 14, 16) < 0.01

    def test_treble_is_untouched_outside_the_sweep(self, house):
        for start, stop in [(0, 10), (20, 30)]:
            assert self.treble_ratio(house, start, stop) == pytest.approx(1.0)

    def test_the_sweep_closes_progressively_then_reopens(self, house):
        ratios = [self.treble_ratio(house, s, s + 1) for s in range(10, 20)]
        centre = int(np.argmin(ratios))
        assert centre in (4, 5)
        closing, opening = ratios[: centre + 1], ratios[centre:]
        assert closing == sorted(closing, reverse=True)
        assert opening == sorted(opening)
        assert ratios[0] > 0.5
        assert ratios[-1] > 0.5

    def test_the_sweep_is_symmetric_about_the_middle(self, house):
        for offset in (1, 2, 3, 4):
            before = self.treble_ratio(house, 15 - offset, 16 - offset)
            after = self.treble_ratio(house, 14 + offset, 15 + offset)
            assert after == pytest.approx(before, abs=0.15)

    def test_the_kick_body_survives_the_filter(self, house):
        # 40 Hz is far below the lowest cutoff of 150 Hz
        centre = slice(14 * self.SR, 16 * self.SR)
        low = np.sqrt((house["wet"][centre] ** 2).mean())
        dry = np.sqrt((house["dry"][centre] ** 2).mean())
        assert low == pytest.approx(dry, rel=0.1)

    def test_cutoff_sweeps_from_12_khz_down_to_150_hz_and_back(self, house):
        cutoff = house["ns"]["cutoff"]

        def at(*times):
            return evaluate(cutoff, np.array(times))

        np.testing.assert_allclose(at(0, 5, 10, 20, 25, 30), 12000, rtol=1e-9)
        assert at(15)[0] == pytest.approx(150)
        # a quarter of the way in the sweep is at half strength, so the cutoff
        # is 12000 * (150 / 12000) ** 0.5
        assert at(12.5)[0] == pytest.approx(12000 / np.sqrt(80))
        np.testing.assert_allclose(at(12), at(18), rtol=1e-9)

    def test_filter_blend_is_zero_outside_and_full_in_the_middle(self, house):
        wet = house["ns"]["wet"]
        out = evaluate(wet, np.array([0, 5, 10, 10.1, 12.5, 15, 17.5, 20, 25, 30]))
        np.testing.assert_allclose(out[[0, 1, 2, 7, 8, 9]], 0.0, atol=1e-12)
        np.testing.assert_allclose(out[[4, 5, 6]], 1.0)
        assert 0 < out[3] < 1

    def test_no_clicks_where_the_filter_enters_or_leaves(self, house):
        for edge in (10, 20):
            around = slice(int((edge - 0.05) * self.SR), int((edge + 0.05) * self.SR))
            wet_step = np.abs(np.diff(house["wet"][around])).max()
            dry_step = np.abs(np.diff(house["dry"][around])).max()
            assert wet_step <= dry_step * 1.5 + 0.01


@pytest.mark.parametrize("name", [n for n in SCRIPTS if n != "classic_synths"])
def test_example_sound_has_not_changed(name, fake_sounddevice, tmp_path, monkeypatch):
    run_example(name, fake_sounddevice, tmp_path, monkeypatch)
    check_golden(name, fake_sounddevice.plays[0][0])


class _SilentSoundDevice:
    @staticmethod
    def play(*args, **kwargs):
        pass

    @staticmethod
    def wait():
        pass


@pytest.fixture(scope="module")
def ode_piece():
    """Compile ode_to_joy once for all the regression tests below"""
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "sounddevice", _SilentSoundDevice)
        ns = runpy.run_path(str(EXAMPLES / "ode_to_joy.py"), run_name="__main__")
    notes, bpm = ns["ode_to_joy_notes"], ns["bpm"]
    total = bpm.time(sum(d for _, d in notes))
    return {
        "notes": notes,
        "bpm": bpm,
        "signal": compile(ns["ode_to_joy"], total),
        "sr": DEFAULT_SAMPLE_RATE,
    }


def note_windows(piece):
    """(note, start, end, next_start) in seconds for each played note"""
    bpm, notes = piece["bpm"], piece["notes"]
    timeline, beat = [], 0.0
    for note, duration in notes:
        timeline.append((note, beat, duration))
        beat += duration
    out = []
    for i, (note, start, duration) in enumerate(timeline):
        if note is None:
            continue
        later = [b for n, b, _ in timeline[i + 1 :] if n is not None]
        next_start = bpm.time(later[0]) if later else None
        out.append((note, bpm.time(start), bpm.time(start + duration), next_start))
    return out


class TestOdeToJoy:
    """Regression tests for the piece that once sounded like noise"""

    def test_signal_never_exceeds_full_scale(self, ode_piece):
        assert np.abs(ode_piece["signal"]).max() <= 1.0

    def test_signal_is_finite(self, ode_piece):
        assert np.isfinite(ode_piece["signal"]).all()

    def test_length_matches_the_score(self, ode_piece):
        total_beats = sum(d for _, d in ode_piece["notes"])
        expected = int(ode_piece["bpm"].time(total_beats) * ode_piece["sr"])
        assert len(ode_piece["signal"]) == expected

    @staticmethod
    def sounding_segment(piece, start, end, next_start):
        """Samples of one note from just after its onset to the end of its
        release, stopping short of any note that follows immediately"""
        stop = end + 0.08
        if next_start is not None:
            stop = min(stop, next_start - 0.005)
        sr = piece["sr"]
        return piece["signal"][int((start + 0.01) * sr) : int(stop * sr)]

    def test_every_note_sounds_at_roughly_its_pitch(self, ode_piece):
        sr = ode_piece["sr"]
        for note, start, end, next_start in note_windows(ode_piece):
            seg = self.sounding_segment(ode_piece, start, end, next_start)
            assert len(seg) / sr > 0.1, "window too short to measure a pitch"
            crossings = np.sum((seg[:-1] < 0) & (seg[1:] >= 0))
            measured = crossings / (len(seg) / sr)
            # The vibrato moves the pitch by up to 5 percent, hence the tolerance
            assert measured == pytest.approx(note.value, rel=0.1), note.name

    def test_notes_are_audible(self, ode_piece):
        for note, start, end, next_start in note_windows(ode_piece):
            seg = self.sounding_segment(ode_piece, start, end, next_start)
            assert np.sqrt((seg**2).mean()) > 0.15, note.name

    def test_rests_are_silent(self, ode_piece):
        sr, signal = ode_piece["sr"], ode_piece["signal"]
        release = 0.1
        checked = 0
        for _, _, end, next_start in note_windows(ode_piece):
            if next_start is None or next_start - end - release < 0.05:
                continue
            lo = int((end + release + 0.01) * sr)
            hi = int((next_start - 0.01) * sr)
            assert np.abs(signal[lo:hi]).max() < 1e-9
            checked += 1
        assert checked > 5, "the score should contain several rests to check"

    def test_note_table_only_uses_known_notes(self, ode_piece):
        for note, duration in ode_piece["notes"]:
            assert note is None or isinstance(note, Note)
            assert duration > 0


@pytest.mark.parametrize("name", SCRIPTS)
def test_importing_an_example_plays_and_saves_nothing(
    name, fake_sounddevice, tmp_path, monkeypatch
):
    # The docs and tests load examples to look at their signals, and nobody
    # wants that to come out of the speakers or litter the disk
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(EXAMPLES / f"{name}.py"))
    assert fake_sounddevice.plays == []
    assert list(tmp_path.iterdir()) == []
