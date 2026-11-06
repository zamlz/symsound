"""Run every example script end to end with audio stubbed out

Examples are the closest thing this project has to acceptance tests: they
exercise the whole pipeline (symbolic construction -> compile -> play/save).
"""

import re
import runpy
import sys
from pathlib import Path

import numpy as np
import pygame
import pytest
import soundfile as sf

from daw import keyboard as keyboard_module
from daw.constants import DEFAULT_SAMPLE_RATE, Note
from daw.live import Mixer, NoteSource
from daw.waves.utils import compile
from golden import check_golden
from helpers import evaluate

pytestmark = pytest.mark.slow

EXAMPLES = Path(__file__).parent.parent / "examples"
SCRIPTS = sorted(p.stem for p in EXAMPLES.glob("*.py"))
# These build signals but never play them when run: tested on their own below
NOT_PLAYED = {"classic_synths", "keyboard_synth"}
# These open a window that stays until a person closes it, so running one as a
# script would never finish: they are run with the window loop replaced below
INTERACTIVE = {"keyboard_synth"}


def run_example(name, fake_sounddevice, tmp_path, monkeypatch):
    """Run an example in a scratch directory and return its namespace"""
    monkeypatch.chdir(tmp_path)
    np.random.seed(0)
    return runpy.run_path(str(EXAMPLES / f"{name}.py"), run_name="__main__")


def test_every_example_that_opens_a_window_is_listed_as_interactive():
    # Running one as a script would block the test suite forever (it once
    # held it up for six minutes), so each must be in INTERACTIVE
    windows = {
        name
        for name in SCRIPTS
        if re.search(r"\b(keyboard|visualize)\(", (EXAMPLES / f"{name}.py").read_text())
    }
    assert windows == INTERACTIVE


def test_every_example_is_covered_by_this_file():
    assert set(SCRIPTS) == {
        "c_major_shimmer",
        "classic_synths",
        "house",
        "house_with_low_pass",
        "keyboard_synth",
        "ode_to_joy",
        "weighted_blend_test",
    }, "add new examples to the tests below"


@pytest.mark.parametrize("name", [n for n in SCRIPTS if n not in INTERACTIVE])
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


@pytest.mark.parametrize("name", [n for n in SCRIPTS if n not in NOT_PLAYED])
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


@pytest.mark.parametrize("name", [n for n in SCRIPTS if n not in NOT_PLAYED])
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
    assert fake_sounddevice.streams == []
    assert list(tmp_path.iterdir()) == []


class TestKeyboardSynth:
    """keyboard_synth.py: an FM bell and a sub, driven through tanh, with a
    tremolo and an ADSR envelope, for the keyboard"""

    SR = 22050

    @pytest.fixture(scope="class")
    @classmethod
    def synth(cls):
        return runpy.run_path(str(EXAMPLES / "keyboard_synth.py"))

    def render(self, synth, note, held=None, seconds=2.0):
        """What the keyboard hears for a note: held for `held` seconds if given"""
        source = NoteSource(synth["synth"](note), self.SR)
        count = round(seconds * self.SR)
        if held is None:
            return source.render(0, count)
        return source.render(0, count, held)

    def reference(self, synth, note, t, held=None):
        """The sound worked out again with NumPy, from the description in the
        example's comments, without SymPy"""
        attack, decay, sustain, release = 0.004, 0.7, 0.3, 0.6
        held = 1e9 if held is None else held

        def level(x):
            return np.where(
                x < attack,
                x / attack,
                np.where(
                    x < attack + decay,
                    1 + (sustain - 1) * (x - attack) / decay,
                    sustain,
                ),
            )

        envelope = np.where(
            t <= held,
            level(t),
            np.where(
                t <= held + release,
                level(held) * (1 - (t - held) / release),
                0.0,
            ),
        )
        index = 0.35 + 3.0 * np.exp(-4.0 * t)
        carrier = 2 * np.pi * note.value * t
        bell = np.sin(carrier + index * np.sin(2 * carrier))
        voice = bell + 0.5 * np.sin(2 * np.pi * (note.value / 2) * t)
        swell = np.minimum(1, t / 1.0)
        tremolo = 1 - 0.18 * swell * (1 - np.sin(2 * np.pi * 5.5 * t)) / 2
        return np.tanh(1.6 * voice) * tremolo * envelope

    @pytest.mark.parametrize("note", [Note.A2, Note.C4, Note.A4, Note.E5])
    def test_it_matches_the_description_sample_for_sample(self, synth, note):
        t = np.arange(round(2.0 * self.SR)) / self.SR
        np.testing.assert_allclose(
            self.render(synth, note), self.reference(synth, note, t), atol=1e-6
        )

    @pytest.mark.parametrize("held", [0.002, 0.3, 1.0])
    def test_a_release_matches_the_description_too(self, synth, held):
        t = np.arange(round(2.0 * self.SR)) / self.SR
        np.testing.assert_allclose(
            self.render(synth, Note.C4, held),
            self.reference(synth, Note.C4, t, held),
            atol=1e-6,
        )

    @pytest.mark.parametrize("note", [Note.A2, Note.A3, Note.A4, Note.E5])
    def test_the_pitch_that_is_heard_is_the_note(self, synth, note):
        heard = self.render(synth, note, seconds=3.0)[self.SR : 3 * self.SR]
        spectrum = np.abs(np.fft.rfft(heard * np.hanning(len(heard))))
        peak = np.argmax(spectrum) * self.SR / len(heard)
        assert peak == pytest.approx(note.value, abs=2.0)

    def test_it_starts_bright_and_mellows_into_a_pure_tone(self, synth):
        heard = self.render(synth, Note.A3, seconds=2.5)

        def centroid(a, b):
            part = heard[int(a * self.SR) : int(b * self.SR)]
            spectrum = np.abs(np.fft.rfft(part * np.hanning(len(part))))
            freqs = np.fft.rfftfreq(len(part), 1 / self.SR)
            return (spectrum * freqs).sum() / spectrum.sum()

        assert centroid(0.01, 0.13) > 1.5 * centroid(2.0, 2.4)

    def test_the_tremolo_wobbles_the_loudness_at_its_rate_and_depth(self, synth):
        # Past the fade-in and the decay, loudness measured over 50 ms windows
        # should rise and fall 5.5 times a second, by about 18% of its level
        heard = self.render(synth, Note.A3, seconds=6.0)
        window, hop = int(0.05 * self.SR), int(0.005 * self.SR)
        starts = range(int(3.0 * self.SR), 6 * self.SR - window, hop)
        loudness = np.array(
            [np.sqrt((heard[a : a + window] ** 2).mean()) for a in starts]
        )
        ripple = np.abs(np.fft.rfft(loudness - loudness.mean()))
        rates = np.fft.rfftfreq(len(loudness), hop / self.SR)
        band = (rates > 2) & (rates < 15)
        assert rates[band][np.argmax(ripple[band])] == pytest.approx(5.5, abs=0.5)
        depth = (loudness.max() - loudness.min()) / (loudness.max() + loudness.min())
        assert depth == pytest.approx(0.1, abs=0.025)

    def test_every_note_stays_inside_full_scale_however_loud(self, synth):
        for note in (Note.A1, Note.C3, Note.C4, Note.G5, Note.C7):
            assert np.abs(self.render(synth, note, seconds=1.0)).max() <= 1.0

    def test_the_envelope_starts_at_silence_sustains_and_dies_after_the_release(
        self, synth
    ):
        held = self.render(synth, Note.C4, held=1.0, seconds=2.5)
        assert held[0] == 0.0
        assert np.abs(held[:5]).max() < 0.2  # the attack ramps up from silence
        assert np.abs(held[int(1.0 * self.SR) : int(1.3 * self.SR)]).max() > 0.05
        assert (held[int(1.0 * self.SR + 0.6 * self.SR) + 5 :] == 0).all()

    def test_the_sound_is_a_signal_in_time_and_the_active_duration_only(self, synth):
        from daw.envelope import ACTIVE_DURATION
        from daw.waves.time import TIME

        expression = synth["synth"](Note.A3)
        assert expression.free_symbols == {TIME, ACTIVE_DURATION}

    def test_every_key_of_the_keyboard_has_an_envelope(self, synth):
        # The keyboard refuses a signal that cannot tell when the key came up
        for note in Note:
            assert NoteSource(synth["synth"](note), self.SR).has_release

    def test_it_is_nothing_like_a_plain_sine(self, synth):
        from daw.waves.basic import sine

        heard = self.render(synth, Note.A3, seconds=1.0)[: self.SR // 2]
        plain = compile(sine(Note.A3) * 0.5, 0.5, self.SR)
        assert np.abs(heard - plain).mean() > 0.05

    def test_a_chord_of_four_keys_stays_inside_full_scale(self, synth):
        mixer = Mixer(gain=0.25)
        for note in (Note.C4, Note.E4, Note.G4, Note.C5):
            mixer.note_on(note, NoteSource(synth["synth"](note), self.SR))
        assert np.abs(mixer.render(2 * self.SR)).max() <= 1.0

    def test_the_sound_has_not_changed(self, synth):
        mixer = Mixer(gain=0.25)
        notes = (Note.C4, Note.E4, Note.G4)
        for note in notes:
            mixer.note_on(note, NoteSource(synth["synth"](note), DEFAULT_SAMPLE_RATE))
        held = mixer.render(round(1.5 * DEFAULT_SAMPLE_RATE))
        for note in notes:
            mixer.note_off(note)
        tail = mixer.render(round(1.0 * DEFAULT_SAMPLE_RATE))
        check_golden("keyboard_synth", np.concatenate([held, tail]))

    def test_run_as_a_script_it_opens_the_keyboard_with_this_sound(
        self, fake_sounddevice, monkeypatch
    ):
        opened = {}

        def fake_run(window, size):
            opened["window"] = window

        monkeypatch.setattr(keyboard_module, "_run", fake_run)
        runpy.run_path(str(EXAMPLES / "keyboard_synth.py"), run_name="__main__")
        window = opened["window"]
        assert window.title == "keyboard_synth.py"
        window.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_h, mod=0))
        stream = fake_sounddevice.streams[0]
        blocks = []
        for _ in range(60):
            outdata = np.zeros((1024, 1), dtype=np.float32)
            stream.callback(outdata, 1024, None, None)
            blocks.append(outdata[:, 0])
        heard = np.concatenate(blocks)[DEFAULT_SAMPLE_RATE // 2 :]
        spectrum = np.abs(np.fft.rfft(heard * np.hanning(len(heard))))
        assert np.argmax(spectrum) * DEFAULT_SAMPLE_RATE / len(heard) == pytest.approx(
            440.0, abs=2.0
        )
