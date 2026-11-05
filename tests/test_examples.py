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
        "kick",
        "ode_to_joy",
        "weighted_blend_test",
    }, "add new examples to the tests below"


@pytest.mark.parametrize("name", SCRIPTS)
def test_example_runs_without_error(name, fake_sounddevice, tmp_path, monkeypatch):
    run_example(name, fake_sounddevice, tmp_path, monkeypatch)


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


def test_kick_writes_its_wav_file(fake_sounddevice, tmp_path, monkeypatch):
    run_example("kick", fake_sounddevice, tmp_path, monkeypatch)
    info = sf.info(tmp_path / "beat.wav")
    assert info.samplerate == DEFAULT_SAMPLE_RATE
    assert info.frames == 30 * DEFAULT_SAMPLE_RATE
    assert info.channels == 1


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
