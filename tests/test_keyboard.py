"""The playable keyboard: the key map, the piano drawing, input and sound

Sound is measured: every key is played and the pitch that comes out is
compared with the note it should be.
"""

import threading

import numpy as np
import pygame
import pytest
from hypothesis import given
from hypothesis import strategies as st

from daw import keyboard as keyboard_module
from daw._draw import _inside
from daw.analysis import LiveSpectrogram
from daw.constants import Note
from daw.envelope import adsr_envelope
from daw.keyboard import (
    BLACK,
    BLACK_KEY,
    KEYS,
    LIT_BLACK,
    LIT_WHITE,
    NAMES,
    NOTES,
    WHITE,
    WHITE_KEY,
    _key_under,
    _Keyboard,
    _layout_for,
    _piano_keys,
    keyboard,
    keyboard_snapshot,
    note_at,
    note_name,
)
from daw.live import Instrument, Mixer, RingBuffer
from daw.waves.basic import sine

RATE = 8000


def voice(note):
    return sine(note) * adsr_envelope(0.01, 0.05, 0.8, 0.1)


@pytest.fixture(scope="module")
def instrument():
    """Compiled once: compiling every note is the slow part"""
    made = Instrument(voice, RATE)
    made.precompile(NOTES)
    return made


@pytest.fixture
def window(instrument):
    return _Keyboard(instrument, Mixer(gain=1.0), RingBuffer(16384))


def key(code):
    return pygame.event.Event(pygame.KEYDOWN, key=code, mod=0)


def key_up(code):
    return pygame.event.Event(pygame.KEYUP, key=code, mod=0)


def click(pos, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button)


def release(pos, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=button)


def move(pos):
    return pygame.event.Event(
        pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(1, 0, 0)
    )


def pitch(samples, rate=RATE):
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return np.argmax(spectrum) * rate / len(samples)


def heard(window, seconds=1.0):
    return window.mixer.render(round(seconds * RATE))


def key_rect(window, semitone):
    return next(
        k.rect for k in _piano_keys(window.layout.piano) if k.semitone == semitone
    )


def centre_of(window, semitone, low=False):
    x, y, w, h = key_rect(window, semitone)
    return (x + w // 2, y + (h - 6 if low else h // 4))


class TestKeyMap:
    def test_there_is_a_key_for_every_semitone_of_one_and_a_half_octaves(self):
        assert sorted(semitone for _, _, semitone in KEYS) == list(range(17))

    def test_no_computer_key_is_used_twice(self):
        keys = [code for code, _, _ in KEYS]
        labels = [label for _, label, _ in KEYS]
        assert len(set(keys)) == len(keys) and len(set(labels)) == len(labels)

    def test_the_labels_match_the_keys(self):
        for code, label, _ in KEYS:
            assert pygame.key.name(code).upper() == label

    def test_the_home_row_is_white_and_the_row_above_is_black(self):
        for _, label, semitone in KEYS:
            assert (label in "WETYUOP") == (semitone in BLACK)

    def test_the_black_keys_follow_a_real_pianos_pattern(self):
        for semitone in range(17):
            assert (semitone in BLACK) == ("#" in NAMES[semitone % 12])

    def test_the_white_keys_run_upwards(self):
        assert WHITE == sorted(WHITE) and len(WHITE) == 10

    def test_the_keys_that_move_the_octave_or_quit_are_not_notes(self):
        used = {code for code, _, _ in KEYS}
        assert not used & {pygame.K_z, pygame.K_x, pygame.K_q, pygame.K_ESCAPE}


class TestNotes:
    def test_known_notes(self):
        assert note_at(4, 0) is Note.C4
        assert note_at(4, 9) is Note.A4
        assert note_at(4, 12) is Note.C5
        assert note_at(1, 0) is Note.C1
        assert note_at(4, 1) is Note.CS4

    def test_a4_is_concert_pitch(self):
        assert note_at(4, 9).value == 440.0

    def test_past_the_top_of_the_table_there_is_no_note(self):
        assert note_at(7, 12) is Note.C8
        assert note_at(7, 13) is None
        assert note_at(7, 16) is None

    @given(st.integers(1, 7), st.integers(0, 16))
    def test_each_step_is_one_semitone_up(self, octave, semitone):
        note = note_at(octave, semitone)
        if note is None:
            return
        c = Note[f"C{octave}"].value
        assert note.value == pytest.approx(c * 2 ** (semitone / 12), rel=0.004)

    @given(st.integers(1, 7), st.integers(0, 16))
    def test_the_name_says_which_note_it_is(self, octave, semitone):
        note = note_at(octave, semitone)
        if note is None:
            return
        expected = f"{NAMES[semitone % 12]}{octave + semitone // 12}"
        assert note_name(note) == expected

    def test_names_are_written_as_a_musician_would(self):
        assert note_name(Note.A4) == "A4"
        assert note_name(Note.CS5) == "C#5"
        assert note_name(Note.A0) == "A0"

    def test_every_note_of_the_table_has_its_own_name(self):
        names = [note_name(note) for note in NOTES]
        assert len(set(names)) == len(NOTES) == 88

    def test_the_table_is_in_rising_order(self):
        values = [note.value for note in NOTES]
        assert values == sorted(values)


class TestPiano:
    RECT = (12, 400, 976, 150)

    def test_ten_white_keys_and_seven_black(self):
        keys = _piano_keys(self.RECT)
        assert sum(not k.black for k in keys) == 10
        assert sum(k.black for k in keys) == 7
        assert {k.semitone for k in keys} == set(range(17))

    def test_the_white_keys_fill_the_piano_without_gaps_or_overlaps(self):
        whites = [k.rect for k in _piano_keys(self.RECT) if not k.black]
        assert whites[0][0] == self.RECT[0]
        for left, right in zip(whites, whites[1:], strict=False):
            assert left[0] + left[2] == right[0]
        last = whites[-1]
        assert last[0] + last[2] == self.RECT[0] + self.RECT[2]
        assert all(w[1] == self.RECT[1] and w[3] == self.RECT[3] for w in whites)

    def test_black_keys_sit_on_the_line_between_their_white_neighbours(self):
        keys = {k.semitone: k for k in _piano_keys(self.RECT)}
        for semitone in BLACK:
            below, above = keys[semitone - 1].rect, keys[semitone + 1].rect
            black = keys[semitone].rect
            assert below[0] < black[0] + black[2] // 2 - 1
            assert black[0] + black[2] // 2 + 1 > above[0] - 3
            seam = above[0]
            assert black[0] < seam < black[0] + black[2]

    def test_black_keys_are_shorter_than_white_keys_and_drawn_last(self):
        keys = _piano_keys(self.RECT)
        flags = [k.black for k in keys]
        assert flags == sorted(flags)
        black = next(k for k in keys if k.black)
        assert black.rect[3] < self.RECT[3]

    def test_the_keys_stay_inside_the_piano(self):
        for k in _piano_keys(self.RECT):
            x, y, w, h = k.rect
            assert self.RECT[0] <= x and x + w <= self.RECT[0] + self.RECT[2]
            assert self.RECT[1] <= y and y + h <= self.RECT[1] + self.RECT[3]

    def test_a_click_high_up_between_two_whites_hits_the_black_key_between_them(self):
        keys = {k.semitone: k.rect for k in _piano_keys(self.RECT)}
        seam = keys[2][0]
        assert _key_under(self.RECT, seam, self.RECT[1] + 10) == 1

    def test_lower_down_the_same_place_hits_a_white_key(self):
        keys = {k.semitone: k.rect for k in _piano_keys(self.RECT)}
        seam = keys[2][0]
        assert _key_under(self.RECT, seam + 2, self.RECT[1] + 140) == 2
        assert _key_under(self.RECT, seam - 2, self.RECT[1] + 140) == 0

    def test_outside_the_piano_there_is_no_key(self):
        assert _key_under(self.RECT, 5, 450) is None
        assert _key_under(self.RECT, 500, 300) is None
        assert _key_under(self.RECT, 500, 560) is None

    @given(st.integers(12, 987), st.integers(400, 549))
    def test_every_point_of_the_piano_is_some_key(self, x, y):
        assert _key_under(self.RECT, x, y) in range(17)

    @given(st.integers(300, 3000), st.integers(300, 2000))
    def test_the_layout_fits_the_window_and_nothing_overlaps(self, width, height):
        layout = _layout_for(width, height)
        rects = [layout.header, layout.scope, layout.spectrogram, layout.piano]
        for x, y, w, h in rects:
            assert x >= 0 and y >= 0 and w > 0 and h > 0
            assert x + w <= max(width, 400) and y + h <= max(height, 380)
        for a in range(len(rects)):
            for b in range(a + 1, len(rects)):
                ax, ay, aw, ah = rects[a]
                bx, by, bw, bh = rects[b]
                assert ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay

    def test_the_piano_is_at_the_bottom_and_the_spectrogram_has_the_most_room(self):
        layout = _layout_for(1000, 640)
        assert layout.header[1] < layout.scope[1] < layout.spectrogram[1]
        assert layout.spectrogram[1] < layout.piano[1]
        assert layout.spectrogram[3] > layout.scope[3]


class TestPlaying:
    def test_a_key_plays_its_note_at_the_right_pitch(self, window):
        window.handle(key(pygame.K_a))
        assert pitch(heard(window)) == pytest.approx(Note.C4.value, abs=1.5)

    @pytest.mark.parametrize("code, label, semitone", KEYS)
    def test_every_key_sounds_at_its_own_pitch(self, window, code, label, semitone):
        window.handle(key(code))
        expected = note_at(4, semitone).value
        assert pitch(heard(window)) == pytest.approx(expected, abs=1.5)

    def test_releasing_a_key_lets_the_note_die_away(self, window):
        window.handle(key(pygame.K_a))
        heard(window, 0.3)
        window.handle(key_up(pygame.K_a))
        tail = heard(window, 0.5)
        assert np.abs(tail[: RATE // 20]).max() > 0.05
        assert (tail[RATE // 4 :] == 0).all()
        heard(window, 0.1)  # a note is dropped at the first chunk that is silent
        assert window.mixer.active == 0
        assert window.sounding == {}

    def test_a_chord_holds_every_note(self, window):
        for code in (pygame.K_a, pygame.K_d, pygame.K_g):
            window.handle(key(code))
        assert window.mixer.active == 3
        spectrum = np.abs(np.fft.rfft(heard(window) * np.hanning(RATE)))
        found = sorted(np.argsort(spectrum)[-3:])
        wanted = [Note.C4.value, Note.E4.value, Note.G4.value]
        for peak, frequency in zip(found, wanted, strict=True):
            assert peak == pytest.approx(frequency, abs=2)

    def test_letting_go_of_one_key_keeps_the_others(self, window):
        window.handle(key(pygame.K_a))
        window.handle(key(pygame.K_d))
        window.handle(key_up(pygame.K_a))
        assert window.mixer.is_held(pygame.K_d) and not window.mixer.is_held(pygame.K_a)

    def test_key_auto_repeat_does_not_restart_the_note(self, window):
        window.handle(key(pygame.K_a))
        first = heard(window, 0.1)
        window.handle(key(pygame.K_a))
        window.handle(key(pygame.K_a))
        assert window.mixer.active == 1
        again = heard(window, 0.1)
        # A restarted note would begin its attack again, far quieter than sustain
        assert np.abs(again).max() > np.abs(first).max() * 0.8

    def test_a_repeat_after_an_octave_change_does_not_rename_the_held_note(
        self, window
    ):
        window.handle(key(pygame.K_a))
        window.handle(key(pygame.K_x))
        window.handle(key(pygame.K_a))  # auto-repeat of the same press
        assert window.sounding[pygame.K_a][0] is Note.C4

    def test_a_key_released_that_was_never_pressed_does_nothing(self, window):
        window.handle(key_up(pygame.K_a))
        assert window.mixer.active == 0

    @pytest.mark.parametrize(
        "code", [pygame.K_LSUPER, pygame.K_F5, pygame.K_1, pygame.K_SPACE, pygame.K_c]
    )
    def test_other_keys_do_nothing_and_do_not_quit(self, window, code):
        window.handle(key(code))
        window.handle(key_up(code))
        assert not window.quit and window.mixer.active == 0

    @pytest.mark.parametrize("code", [pygame.K_q, pygame.K_ESCAPE])
    def test_q_and_escape_quit(self, window, code):
        window.handle(key(code))
        assert window.quit

    def test_closing_the_window_quits(self, window):
        window.handle(pygame.event.Event(pygame.QUIT))
        assert window.quit

    def test_losing_focus_releases_every_held_key(self, window):
        window.handle(key(pygame.K_a))
        window.handle(key(pygame.K_d))
        window.handle(pygame.event.Event(pygame.WINDOWFOCUSLOST))
        assert window.sounding == {}
        assert not window.mixer.is_held(pygame.K_a)
        assert not window.mixer.is_held(pygame.K_d)

    def test_resizing_moves_the_piano(self, window):
        window.handle(pygame.event.Event(pygame.VIDEORESIZE, w=700, h=500))
        assert window.layout == _layout_for(700, 500)


class TestOctaves:
    def test_z_and_x_move_the_octave_and_the_pitch_follows(self, window):
        window.handle(key(pygame.K_x))
        assert window.octave == 5
        window.handle(key(pygame.K_a))
        assert pitch(heard(window)) == pytest.approx(Note.C5.value, abs=1.5)

    def test_down_an_octave_halves_the_pitch(self, window):
        window.handle(key(pygame.K_z))
        window.handle(key(pygame.K_a))
        assert pitch(heard(window, 2.0)) == pytest.approx(Note.C3.value, abs=1.0)

    def test_the_octave_is_kept_between_the_ends(self, window):
        for _ in range(20):
            window.handle(key(pygame.K_z))
        assert window.octave == 1
        for _ in range(20):
            window.handle(key(pygame.K_x))
        assert window.octave == 7

    def test_a_held_note_keeps_its_pitch_when_the_octave_changes(self, window):
        window.handle(key(pygame.K_a))
        window.handle(key(pygame.K_x))
        assert pitch(heard(window)) == pytest.approx(Note.C4.value, abs=1.5)
        window.handle(key_up(pygame.K_a))
        heard(window, 0.5)
        heard(window, 0.1)
        assert window.mixer.active == 0

    def test_notes_beyond_the_table_are_ignored(self, window):
        for _ in range(5):
            window.handle(key(pygame.K_x))
        window.handle(key(pygame.K_SEMICOLON))
        assert window.mixer.active == 0 and window.sounding == {}

    def test_the_starting_octave_can_be_chosen(self, instrument):
        started = _Keyboard(instrument, Mixer(1.0), RingBuffer(4096), octave=2)
        started.handle(key(pygame.K_a))
        assert pitch(started.mixer.render(RATE)) == pytest.approx(Note.C2.value, abs=1)

    @pytest.mark.parametrize("octave", [0, 8, -1])
    def test_a_starting_octave_off_the_table_is_refused(self, instrument, octave):
        with pytest.raises(ValueError, match="octave"):
            _Keyboard(instrument, Mixer(), RingBuffer(4096), octave=octave)


class TestMouse:
    def test_clicking_a_white_key_plays_it_and_letting_go_stops_it(self, window):
        window.handle(click(centre_of(window, 7, low=True)))
        assert pitch(heard(window)) == pytest.approx(Note.G4.value, abs=1.5)
        window.handle(release((0, 0)))
        heard(window, 0.5)
        heard(window, 0.1)
        assert window.mixer.active == 0

    def test_clicking_a_black_key_plays_the_sharp(self, window):
        window.handle(click(centre_of(window, 1)))
        assert pitch(heard(window)) == pytest.approx(Note.CS4.value, abs=1.5)

    def test_dragging_across_the_piano_slides_from_note_to_note(self, window):
        window.handle(click(centre_of(window, 0, low=True)))
        window.handle(move(centre_of(window, 2, low=True)))
        assert [n for n, _ in window.sounding.values()] == [Note.D4]
        window.handle(move(centre_of(window, 2, low=True)))  # same key: no change
        assert window.mixer.active == 2  # the first note is dying away
        window.handle(release((0, 0)))
        assert window.sounding == {}

    def test_dragging_off_the_piano_stops_the_note(self, window):
        window.handle(click(centre_of(window, 0, low=True)))
        window.handle(move((500, 100)))
        assert window.sounding == {}

    def test_moving_the_mouse_over_the_piano_without_clicking_is_silent(self, window):
        window.handle(move(centre_of(window, 4, low=True)))
        assert window.mixer.active == 0

    def test_clicking_outside_the_piano_does_nothing(self, window):
        window.handle(click((500, 300)))
        window.handle(click((0, 0)))
        assert window.mixer.active == 0

    def test_only_the_left_button_plays(self, window):
        window.handle(click(centre_of(window, 0, low=True), button=3))
        assert window.mixer.active == 0
        window.handle(click(centre_of(window, 0, low=True)))
        window.handle(release((0, 0), button=3))
        assert window.sounding != {}

    def test_the_mouse_and_the_keyboard_play_together(self, window):
        window.handle(key(pygame.K_a))
        window.handle(click(centre_of(window, 7, low=True)))
        assert window.mixer.active == 2
        window.handle(release((0, 0)))
        assert window.mixer.is_held(pygame.K_a)
        assert not window.mixer.is_held("mouse")


class TestDisplay:
    def test_without_audio_the_display_makes_the_sound_itself(self, instrument):
        silent = _Keyboard(instrument, Mixer(1.0), RingBuffer(16384), drive_clock=True)
        silent.handle(key(pygame.K_a))
        silent.advance(0.5)
        assert silent.ring.written == RATE // 2
        assert pitch(silent.ring.latest(RATE // 2)) == pytest.approx(
            Note.C4.value, abs=2
        )

    def test_with_audio_the_display_only_watches(self, window):
        window.handle(key(pygame.K_a))
        window.advance(0.5)
        assert window.ring.written == 0

    def test_the_spectrogram_takes_in_what_was_played(self, window):
        window.handle(key(pygame.K_a))
        window.ring.write(window.mixer.render(RATE))
        window.advance(0)
        assert len(window._spectrogram.levels) == RATE // 512

    def test_the_picture_of_a_held_note_has_a_lit_key_and_a_bright_spectrogram(
        self, window
    ):
        window.handle(key(pygame.K_a))
        window.ring.write(window.mixer.render(RATE))
        window.advance(0)
        surface = pygame.Surface((1000, 640))
        window.draw(surface)
        x, y, w, h = key_rect(window, 0)
        assert tuple(surface.get_at((x + w // 2, y + h - 3)))[:3] == LIT_WHITE
        d = key_rect(window, 2)
        assert (
            tuple(surface.get_at((d[0] + d[2] // 2, d[1] + d[3] - 3)))[:3] == WHITE_KEY
        )
        sx, sy, sw, sh = window.layout.spectrogram
        area = pygame.surfarray.array3d(surface)[
            sx + 120 : sx + sw - 3, sy + 20 : sy + sh
        ]
        assert area.max() > 200

    def test_a_held_black_key_lights_up_and_the_others_do_not(self, window):
        window.handle(key(pygame.K_w))
        surface = pygame.Surface((1000, 640))
        window.draw(surface)
        x, y, w, h = key_rect(window, 1)
        assert tuple(surface.get_at((x + w // 2, y + 8)))[:3] == LIT_BLACK
        e = key_rect(window, 3)
        assert tuple(surface.get_at((e[0] + e[2] // 2, e[1] + 8)))[:3] == BLACK_KEY

    def test_a_key_goes_dark_again_when_released(self, window):
        window.handle(key(pygame.K_a))
        window.handle(key_up(pygame.K_a))
        surface = pygame.Surface((1000, 640))
        window.draw(surface)
        x, y, w, h = key_rect(window, 0)
        assert tuple(surface.get_at((x + w // 2, y + h - 3)))[:3] == WHITE_KEY

    def test_the_scope_shows_the_wave_that_is_playing(self, window):
        window.handle(key(pygame.K_a))
        window.ring.write(window.mixer.render(RATE))
        window.advance(0)
        surface = pygame.Surface((1000, 640))
        window.draw(surface)
        x, y, w, h = window.layout.scope
        area = pygame.surfarray.array3d(surface)[x : x + w, y : y + h]
        trace = (area == np.array((110, 200, 255))).all(axis=2)
        assert np.flatnonzero(trace.any(axis=0)).size > h * 0.5

    def test_the_scope_is_flat_when_nothing_plays(self, window):
        surface = pygame.Surface((1000, 640))
        window.draw(surface)
        x, y, w, h = window.layout.scope
        area = pygame.surfarray.array3d(surface)[x : x + w, y : y + h]
        trace = (area == np.array((110, 200, 255))).all(axis=2)
        assert np.flatnonzero(trace.any(axis=0)).size <= 2

    def test_changing_the_octave_changes_the_note_names_on_the_piano(self, window):
        a, b = pygame.Surface((1000, 640)), pygame.Surface((1000, 640))
        window.draw(a)
        window.handle(key(pygame.K_x))
        window.draw(b)
        piano = window.layout.piano
        left = pygame.surfarray.array3d(a)[
            piano[0] : piano[0] + piano[2], piano[1] : piano[1] + piano[3]
        ]
        right = pygame.surfarray.array3d(b)[
            piano[0] : piano[0] + piano[2], piano[1] : piano[1] + piano[3]
        ]
        assert not np.array_equal(left, right)

    @pytest.mark.parametrize("size", [(1000, 640), (400, 380), (1600, 900), (20, 20)])
    def test_it_draws_at_any_size(self, window, size):
        surface = pygame.Surface(size)
        window.draw(surface)
        assert surface.get_size() == size

    def test_the_notes_that_are_held_are_named_in_the_header(self, window):
        a, b = pygame.Surface((1000, 640)), pygame.Surface((1000, 640))
        window.draw(a)
        window.handle(key(pygame.K_a))
        window.draw(b)
        x, y, w, h = window.layout.header
        before = pygame.surfarray.array3d(a)[x + w - 200 : x + w, y : y + 20]
        after = pygame.surfarray.array3d(b)[x + w - 200 : x + w, y : y + 20]
        assert not np.array_equal(before, after)


class TestKeyboardFunction:
    @pytest.fixture
    def opened(self, monkeypatch):
        seen = {}

        def fake_run(window, size):
            seen["window"] = window
            seen["size"] = size

        monkeypatch.setattr(keyboard_module, "_run", fake_run)
        return seen

    def test_every_note_is_compiled_before_the_window_opens(
        self, fake_sounddevice, opened
    ):
        calls = []

        def function(note):
            calls.append(note)
            return sine(note)

        keyboard(function, sample_rate=RATE, require_envelope=False)
        assert sorted(calls, key=lambda n: n.value) == NOTES
        assert opened["window"].instrument.sample_rate == RATE

    def test_the_speakers_play_what_the_keys_ask_for(self, fake_sounddevice, opened):
        keyboard(voice, sample_rate=RATE, gain=1.0)
        window = opened["window"]
        window.handle(key(pygame.K_h))  # A4
        stream = fake_sounddevice.streams[0]
        blocks = []
        for _ in range(16):
            outdata = np.zeros((500, 1), dtype=np.float32)
            stream.callback(outdata, 500, None, None)
            blocks.append(outdata[:, 0])
        assert pitch(np.concatenate(blocks)) == pytest.approx(440.0, abs=2)

    def test_what_the_speakers_play_is_copied_to_the_display(
        self, fake_sounddevice, opened
    ):
        keyboard(voice, sample_rate=RATE)
        window = opened["window"]
        window.handle(key(pygame.K_a))
        outdata = np.zeros((400, 1), dtype=np.float32)
        fake_sounddevice.streams[0].callback(outdata, 400, None, None)
        assert window.ring.written == 400
        np.testing.assert_allclose(window.ring.latest(400), outdata[:, 0], atol=1e-6)

    def test_the_stream_runs_while_the_window_is_open_and_stops_after(
        self, fake_sounddevice, opened
    ):
        keyboard(voice, sample_rate=RATE)
        (stream,) = fake_sounddevice.streams
        assert stream.events == ["start", "stop", "close"]
        assert stream.options["samplerate"] == RATE

    def test_silent_makes_no_stream_and_runs_the_picture_on_the_clock(
        self, fake_sounddevice, opened
    ):
        keyboard(voice, sample_rate=RATE, silent=True)
        assert fake_sounddevice.streams == []
        assert opened["window"].drive_clock

    def test_the_options_reach_the_window(self, fake_sounddevice, opened):
        keyboard(
            voice, sample_rate=RATE, octave=3, gain=0.2, title="mine", size=(700, 500)
        )
        window = opened["window"]
        assert window.octave == 3 and window.title == "mine"
        assert window.mixer.gain == 0.2
        assert opened["size"] == (700, 500)

    def test_the_default_gain_leaves_room_for_a_chord(self, fake_sounddevice, opened):
        keyboard(voice, sample_rate=RATE)
        window = opened["window"]
        for code in (pygame.K_a, pygame.K_d, pygame.K_g, pygame.K_k):
            window.handle(key(code))
        assert np.abs(window.mixer.render(RATE)).max() <= 1.0

    def test_a_signal_with_no_envelope_is_refused_by_default(
        self, fake_sounddevice, opened
    ):
        with pytest.raises(ValueError, match=r"no envelope.*adsr_envelope"):
            keyboard(lambda note: sine(note), sample_rate=RATE)
        assert fake_sounddevice.streams == [] and not opened

    def test_the_error_names_the_note_and_says_how_to_opt_out(
        self, fake_sounddevice, opened
    ):
        with pytest.raises(ValueError) as error:
            keyboard(lambda note: sine(note), sample_rate=RATE)
        assert "A0" in str(error.value)
        assert "require_envelope=False" in str(error.value)

    def test_one_note_without_an_envelope_is_enough_to_refuse(
        self, fake_sounddevice, opened
    ):
        def patchy(note):
            return sine(note) if note is Note.E5 else voice(note)

        with pytest.raises(ValueError, match="E5"):
            keyboard(patchy, sample_rate=RATE)

    def test_a_signal_with_no_envelope_plays_when_asked(self, fake_sounddevice, opened):
        keyboard(lambda note: sine(note), sample_rate=RATE, require_envelope=False)
        window = opened["window"]
        window.handle(key(pygame.K_a))
        window.handle(key_up(pygame.K_a))
        window.mixer.render(RATE // 10)  # the 10 ms fade runs out
        window.mixer.render(RATE // 10)
        assert window.mixer.active == 0

    def test_a_signal_with_an_envelope_needs_no_flag(self, fake_sounddevice, opened):
        keyboard(voice, sample_rate=RATE)
        assert opened["window"].instrument.sample_rate == RATE

    def test_a_filtered_signal_is_refused_before_anything_opens(
        self, fake_sounddevice, opened
    ):
        from daw.filters import low_pass_filter

        with pytest.raises(ValueError, match="filtered"):
            keyboard(lambda note: low_pass_filter(sine(note), 500), sample_rate=RATE)
        assert fake_sounddevice.streams == [] and not opened

    def test_the_stream_is_stopped_even_if_the_window_fails(
        self, fake_sounddevice, monkeypatch
    ):
        def broken(window, size):
            raise RuntimeError("boom")

        monkeypatch.setattr(keyboard_module, "_run", broken)
        with pytest.raises(RuntimeError):
            keyboard(voice, sample_rate=RATE)
        assert fake_sounddevice.streams[0].events == ["start", "stop", "close"]

    def test_the_whole_thing_runs_in_a_real_pygame_window(self, fake_sounddevice):
        timer = threading.Timer(0.5, lambda: pygame.event.post(key(pygame.K_q)))
        timer.start()
        try:
            keyboard(voice, sample_rate=RATE, silent=True, size=(500, 400))
        finally:
            timer.cancel()
        assert not pygame.get_init()


class TestSnapshot:
    def test_a_signal_with_no_envelope_is_refused_by_default(self, tmp_path):
        with pytest.raises(ValueError, match="no envelope"):
            keyboard_snapshot(
                lambda note: sine(note),
                (Note.A4,),
                tmp_path / "x.png",
                sample_rate=RATE,
            )
        assert not (tmp_path / "x.png").exists()

    def test_it_can_draw_one_anyway_when_asked(self, tmp_path):
        keyboard_snapshot(
            lambda note: sine(note),
            (Note.A4,),
            tmp_path / "x.png",
            seconds=0.2,
            sample_rate=RATE,
            require_envelope=False,
        )
        assert (tmp_path / "x.png").exists()

    def test_only_the_held_notes_are_checked(self, tmp_path):
        def patchy(note):
            return voice(note) if note is Note.C4 else sine(note)

        keyboard_snapshot(
            patchy, (Note.C4,), tmp_path / "x.png", seconds=0.2, sample_rate=RATE
        )
        assert (tmp_path / "x.png").exists()

    def test_it_draws_the_held_keys_lit(self, tmp_path):
        out = tmp_path / "deep" / "kb.png"
        keyboard_snapshot(
            voice,
            (Note.C4, Note.E4),
            out,
            seconds=0.5,
            sample_rate=RATE,
            size=(800, 560),
        )
        image = pygame.image.load(str(out))
        assert image.get_size() == (800, 560)
        layout = _layout_for(800, 560)
        keys = {k.semitone: k.rect for k in _piano_keys(layout.piano)}
        for semitone, colour in ((0, LIT_WHITE), (4, LIT_WHITE), (2, WHITE_KEY)):
            x, y, w, h = keys[semitone]
            assert tuple(image.get_at((x + w // 2, y + h - 3)))[:3] == colour

    def test_the_spectrogram_shows_the_chord(self, tmp_path):
        out = tmp_path / "kb.png"
        keyboard_snapshot(voice, (Note.A4,), out, seconds=1.0, sample_rate=RATE)
        image = pygame.image.load(str(out))
        layout = _layout_for(1000, 640)
        x, y, w, h = layout.spectrogram
        area = pygame.surfarray.array3d(image)[x + 120 : x + w - 3, y + 20 : y + h]
        assert area.max() > 200

    def test_with_nothing_held_nothing_is_lit(self, tmp_path):
        out = tmp_path / "kb.png"
        keyboard_snapshot(voice, (), out, seconds=0.2, sample_rate=RATE)
        image = pygame.image.load(str(out))
        layout = _layout_for(1000, 640)
        for key_ in _piano_keys(layout.piano):
            x, y, w, h = key_.rect
            colour = tuple(
                image.get_at((x + w // 2, y + (8 if key_.black else h - 3)))
            )[:3]
            assert colour in (WHITE_KEY, BLACK_KEY)

    def test_the_octave_decides_which_notes_the_held_keys_are(self, tmp_path):
        keyboard_snapshot(
            voice,
            (Note.C5,),
            tmp_path / "a.png",
            octave=5,
            seconds=0.2,
            sample_rate=RATE,
        )
        image = pygame.image.load(str(tmp_path / "a.png"))
        layout = _layout_for(1000, 640)
        x, y, w, h = next(k.rect for k in _piano_keys(layout.piano) if k.semitone == 0)
        assert tuple(image.get_at((x + w // 2, y + h - 3)))[:3] == LIT_WHITE

    def test_the_picture_matches_what_the_window_draws(self, tmp_path, instrument):
        out = tmp_path / "kb.png"
        keyboard_snapshot(
            voice, (Note.G4,), out, seconds=0.3, sample_rate=RATE, size=(700, 500)
        )
        saved = pygame.surfarray.array3d(pygame.image.load(str(out)))
        assert saved.shape[:2] == (700, 500)
        assert saved.std() > 5


class TestLiveSpectrogramInTheWindow:
    def test_the_levels_fit_the_window_history(self, window):
        assert isinstance(window._spectrogram, LiveSpectrogram)
        assert window._spectrogram.sample_rate == RATE


def test_the_window_geometry_helper_matches_inside():
    rect = (10, 10, 20, 20)
    assert _inside(rect, 10, 10) and not _inside(rect, 30, 30)
