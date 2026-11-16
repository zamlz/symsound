# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""The visualizer: its layout, its input handling and what it draws

Drawing goes onto off-screen surfaces, so the tests read the pixels back and
check what is where, not just that something ran.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pygame
import pytest
import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

import symsound
from symsound import visualizer
from symsound._draw import (
    MARGIN,
    PIXELS_PER_COLUMN,
    PLAYED,
    PLAYHEAD,
    SCOPE_COLOUR,
    SPECTROGRAM_ROWS,
    _colourise,
    _inside,
    _run,
)
from symsound.analysis import HOP, row_of_frequency
from symsound.live import Player
from symsound.visualizer import (
    _format_time,
    _layout_for,
    _seek_fraction,
    _Viewer,
    snapshot,
    visualize,
)
from symsound.waves.basic import sine
from symsound.waves.time import TIME
from symsound.waves.utils import compile

RATE = 8000
SECONDS = 4.0


def make_player(expression=None, seconds=SECONDS):
    samples = compile(sine(1000) if expression is None else expression, seconds, RATE)
    return Player(samples, RATE)


def key(code, mod=0):
    return pygame.event.Event(pygame.KEYDOWN, key=code, mod=mod)


def click(pos, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button)


def release(pos, button=1):
    return pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=button)


def move(pos):
    return pygame.event.Event(
        pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(1, 0, 0)
    )


def centre(rect):
    x, y, w, h = rect
    return (x + w // 2, y + h // 2)


def count_pixels(surface, colour, rect):
    x, y, w, h = rect
    area = pygame.surfarray.array3d(surface)[x : x + w, y : y + h]
    return int((area == np.array(colour)).all(axis=2).sum())


class TestFormatTime:
    @pytest.mark.parametrize(
        "seconds, text",
        [
            (0, "0:00.0"),
            (3.25, "0:03.2"),
            (59.99, "0:59.9"),
            (60, "1:00.0"),
            (83.25, "1:23.2"),
            (600, "10:00.0"),
            (-5, "0:00.0"),
        ],
    )
    def test_known_values(self, seconds, text):
        assert _format_time(seconds) == text

    @given(st.floats(min_value=0, max_value=36000))
    def test_it_always_reads_back_to_the_time_within_a_tenth(self, seconds):
        minutes, rest = _format_time(seconds).split(":")
        assert float(minutes) * 60 + float(rest) == pytest.approx(seconds, abs=0.1001)


class TestColourise:
    def test_the_bottom_of_the_scale_is_black_and_the_top_is_bright(self):
        assert _colourise(0.0).tolist() == [0, 0, 3]
        assert _colourise(1.0).sum() > 600

    def test_it_adds_a_colour_axis(self):
        assert _colourise(np.zeros((5, 7))).shape == (5, 7, 3)
        assert _colourise(np.zeros((5, 7))).dtype == np.uint8

    def test_values_outside_the_scale_are_clipped(self):
        assert _colourise(-3).tolist() == _colourise(0).tolist()
        assert _colourise(9).tolist() == _colourise(1).tolist()

    def test_louder_is_brighter(self):
        levels = np.linspace(0, 1, 64)
        brightness = _colourise(levels).astype(int).sum(axis=1)
        assert (np.diff(brightness) >= -8).all()
        assert brightness[-1] > brightness[0] + 500


class TestLayout:
    @given(st.integers(0, 3000), st.integers(0, 2000))
    def test_every_part_fits_inside_the_window_and_none_overlap(self, width, height):
        layout = _layout_for(width, height)
        width, height = max(width, 320), max(height, 260)
        rects = [
            layout.play,
            layout.replay,
            layout.header,
            layout.scope,
            layout.spectrogram,
            layout.seek,
        ]
        for x, y, w, h in rects:
            assert x >= 0 and y >= 0 and w > 0 and h > 0
            assert x + w <= width and y + h <= height
        for a in range(len(rects)):
            for b in range(a + 1, len(rects)):
                ax, ay, aw, ah = rects[a]
                bx, by, bw, bh = rects[b]
                apart = ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay
                assert apart, (a, b, rects)

    def test_the_parts_are_stacked_in_reading_order(self):
        layout = _layout_for(1000, 640)
        assert layout.play[1] < layout.scope[1] < layout.spectrogram[1] < layout.seek[1]
        assert layout.play[0] < layout.replay[0] < layout.header[0]

    def test_the_wide_parts_span_the_window(self):
        layout = _layout_for(1000, 640)
        for rect in (layout.scope, layout.spectrogram, layout.seek):
            assert rect[0] == MARGIN
            assert rect[2] == 1000 - 2 * MARGIN

    def test_the_spectrogram_gets_the_most_room(self):
        layout = _layout_for(1000, 640)
        assert layout.spectrogram[3] > layout.scope[3] > 40

    def test_a_tiny_window_is_laid_out_as_the_smallest_one(self):
        assert _layout_for(10, 10) == _layout_for(320, 260)

    def test_inside_includes_the_top_left_and_excludes_the_far_edge(self):
        rect = (10, 20, 30, 40)
        assert _inside(rect, 10, 20)
        assert _inside(rect, 39, 59)
        assert not _inside(rect, 40, 30)
        assert not _inside(rect, 20, 60)
        assert not _inside(rect, 9, 30)

    def test_seek_fraction_runs_from_zero_to_one_and_is_clamped(self):
        rect = (10, 0, 200, 10)
        assert _seek_fraction(rect, 10) == 0.0
        assert _seek_fraction(rect, 110) == 0.5
        assert _seek_fraction(rect, 210) == 1.0
        assert _seek_fraction(rect, -50) == 0.0
        assert _seek_fraction(rect, 999) == 1.0


class TestKeyboard:
    def setup_method(self):
        self.player = make_player()
        self.viewer = _Viewer(self.player)

    def test_space_plays_and_pauses(self):
        self.viewer.handle(key(pygame.K_SPACE))
        assert self.player.playing
        self.viewer.handle(key(pygame.K_SPACE))
        assert not self.player.playing

    def test_r_replays_from_the_start(self):
        self.player.seek(3.0)
        self.viewer.handle(key(pygame.K_r))
        assert self.player.position == 0 and self.player.playing

    def test_arrows_step_by_a_second_and_shift_by_five(self):
        self.player.seek(2.0)
        self.viewer.handle(key(pygame.K_RIGHT))
        assert self.player.seconds == 3.0
        self.viewer.handle(key(pygame.K_LEFT))
        self.viewer.handle(key(pygame.K_LEFT))
        assert self.player.seconds == 1.0
        self.viewer.handle(key(pygame.K_RIGHT, pygame.KMOD_SHIFT))
        assert self.player.seconds == 4.0  # clamped to the end

    def test_arrows_are_kept_inside_the_signal(self):
        self.viewer.handle(key(pygame.K_LEFT, pygame.KMOD_SHIFT))
        assert self.player.position == 0
        self.player.seek(99)
        self.viewer.handle(key(pygame.K_RIGHT))
        assert self.player.seconds == SECONDS

    def test_home_goes_to_the_start_without_playing(self):
        self.player.seek(2.0)
        self.viewer.handle(key(pygame.K_HOME))
        assert self.player.position == 0 and not self.player.playing

    @pytest.mark.parametrize("code", [pygame.K_q, pygame.K_ESCAPE])
    def test_q_and_escape_quit(self, code):
        assert not self.viewer.quit
        self.viewer.handle(key(code))
        assert self.viewer.quit

    def test_closing_the_window_quits(self):
        self.viewer.handle(pygame.event.Event(pygame.QUIT))
        assert self.viewer.quit

    @pytest.mark.parametrize("code", [pygame.K_a, pygame.K_LSUPER, pygame.K_F5])
    def test_other_keys_do_nothing_and_above_all_do_not_crash(self, code):
        self.viewer.handle(key(code))
        assert not self.viewer.quit
        assert self.player.position == 0 and not self.player.playing

    def test_unrelated_events_are_ignored(self):
        self.viewer.handle(pygame.event.Event(pygame.KEYUP, key=pygame.K_SPACE, mod=0))
        self.viewer.handle(pygame.event.Event(pygame.WINDOWFOCUSGAINED))
        assert not self.player.playing and not self.viewer.quit


class TestMouse:
    def setup_method(self):
        self.player = make_player()
        self.viewer = _Viewer(self.player)
        self.layout = self.viewer.layout

    def test_the_play_button_toggles(self):
        self.viewer.handle(click(centre(self.layout.play)))
        assert self.player.playing
        self.viewer.handle(click(centre(self.layout.play)))
        assert not self.player.playing

    def test_the_replay_button_restarts(self):
        self.player.seek(3.0)
        self.viewer.handle(click(centre(self.layout.replay)))
        assert self.player.position == 0 and self.player.playing

    def test_clicking_the_seek_bar_jumps_there(self):
        x, y, w, h = self.layout.seek
        self.viewer.handle(click((x + w // 4, y + h // 2)))
        assert self.player.seconds == pytest.approx(SECONDS / 4, abs=0.02)

    def test_the_far_ends_of_the_seek_bar_are_the_start_and_the_end(self):
        x, y, w, h = self.layout.seek
        self.viewer.handle(click((x + w - 1, y + 1)))
        assert self.player.seconds == pytest.approx(SECONDS, abs=0.02)
        self.viewer.handle(click((x, y + 1)))
        assert self.player.position == 0

    def test_dragging_scrubs_pauses_while_held_and_resumes_after(self):
        x, y, w, h = self.layout.seek
        self.player.play()
        self.viewer.handle(click((x + w // 10, y + 5)))
        assert not self.player.playing
        self.viewer.handle(move((x + w // 2, y + 5)))
        assert self.player.seconds == pytest.approx(SECONDS / 2, abs=0.02)
        self.viewer.handle(move((x + 10 * w, y + 5)))  # dragged off the end
        assert self.player.seconds == SECONDS
        self.viewer.handle(release((x + w - 1, y + 5)))
        assert self.player.playing or self.player.finished

    def test_a_drag_that_began_paused_ends_paused(self):
        x, y, w, h = self.layout.seek
        self.viewer.handle(click((x + w // 3, y + 5)))
        self.viewer.handle(release((x + w // 3, y + 5)))
        assert not self.player.playing

    def test_moving_the_mouse_without_a_drag_does_nothing(self):
        x, y, w, h = self.layout.seek
        self.viewer.handle(move((x + w // 2, y + 5)))
        assert self.player.position == 0

    def test_releasing_without_a_drag_does_not_start_playback(self):
        self.viewer.handle(release((5, 5)))
        assert not self.player.playing

    def test_only_the_left_button_counts(self):
        self.viewer.handle(click(centre(self.layout.play), button=3))
        assert not self.player.playing
        x, y, w, h = self.layout.seek
        self.viewer.handle(click((x + w // 2, y + 5), button=2))
        assert self.player.position == 0
        self.viewer.handle(click((x + w // 2, y + 5)))
        self.viewer.handle(release((x, y), button=3))
        assert self.viewer._dragging

    def test_clicking_elsewhere_does_nothing(self):
        self.viewer.handle(click(centre(self.layout.spectrogram)))
        self.viewer.handle(click((0, 0)))
        assert self.player.position == 0 and not self.player.playing

    def test_resizing_the_window_moves_the_buttons_and_the_bar(self):
        self.viewer.handle(pygame.event.Event(pygame.VIDEORESIZE, w=600, h=400))
        assert self.viewer.layout == _layout_for(600, 400)
        x, y, w, h = self.viewer.layout.seek
        self.viewer.handle(click((x + w // 2, y + 5)))
        assert self.player.seconds == pytest.approx(SECONDS / 2, abs=0.03)


class TestClock:
    def test_without_audio_the_display_moves_the_playhead(self):
        player = make_player()
        viewer = _Viewer(player, drive_clock=True)
        player.play()
        viewer.advance(0.5)
        assert player.seconds == pytest.approx(0.5, abs=1 / RATE)

    def test_with_audio_the_display_leaves_it_alone(self):
        player = make_player()
        viewer = _Viewer(player)
        player.play()
        viewer.advance(0.5)
        assert player.position == 0

    def test_paused_it_does_not_move_either_way(self):
        player = make_player()
        viewer = _Viewer(player, drive_clock=True)
        viewer.advance(0.5)
        assert player.position == 0


class TestDrawing:
    def draw(self, expression=None, at=2.0, size=(1000, 640), playing=False, **options):
        player = make_player(expression)
        player.seek(at)
        if playing:
            player.play()
        viewer = _Viewer(player, **options)
        surface = pygame.Surface(size)
        viewer.draw(surface)
        return viewer, surface

    @pytest.mark.parametrize("size", [(1000, 640), (320, 260), (1600, 900), (50, 50)])
    def test_it_draws_at_any_window_size(self, size):
        _, surface = self.draw(size=size)
        assert surface.get_size() == size
        assert pygame.surfarray.array3d(surface).std() > 5

    def test_drawing_follows_the_surface_it_is_given(self):
        viewer, _ = self.draw(size=(700, 500))
        assert viewer.layout == _layout_for(700, 500)

    @pytest.mark.parametrize("at", [0.0, 1.0, SECONDS])
    def test_it_draws_at_the_start_the_middle_and_the_very_end(self, at):
        self.draw(at=at)

    def test_the_playhead_on_the_seek_bar_is_where_the_time_is(self):
        for fraction in (0.0, 0.25, 0.5, 0.9):
            viewer, surface = self.draw(at=fraction * SECONDS)
            x, y, w, h = viewer.layout.seek
            row = pygame.surfarray.array3d(surface)[x : x + w, y + h - 3]
            found = np.flatnonzero((row == np.array(PLAYHEAD)).all(axis=1))
            assert found.size
            assert found.mean() / w == pytest.approx(fraction, abs=0.01)

    def test_the_seek_bar_shows_played_and_unplayed_parts_in_different_colours(self):
        viewer, surface = self.draw(at=SECONDS / 2)
        x, y, w, h = viewer.layout.seek
        left = (x, y, w // 2 - 3, h)
        right = (x + w // 2 + 3, y, w // 2 - 3, h)
        assert count_pixels(surface, PLAYED, left) > 100
        assert count_pixels(surface, PLAYED, right) == 0

    def test_the_seek_bar_outlines_loud_and_quiet_parts(self):
        fade = sine(1000) * sp.Piecewise((1.0, TIME < SECONDS / 2), (0.1, True))
        viewer, surface = self.draw(fade, at=SECONDS)
        x, y, w, h = viewer.layout.seek
        loud = count_pixels(surface, PLAYED, (x, y, w // 2 - 3, h))
        quiet = count_pixels(surface, PLAYED, (x + w // 2 + 3, y, w // 2 - 3, h))
        assert loud > 3 * quiet > 0

    def test_the_scope_draws_the_wave_and_silence_is_a_flat_line(self):

        loud, surface = self.draw(sine(200) * 0.9, at=2.0)
        flat, quiet = self.draw(sp.Float(0.0), at=2.0)
        x, y, w, h = loud.layout.scope
        loud_rows = pygame.surfarray.array3d(surface)[x : x + w, y : y + h]
        flat_rows = pygame.surfarray.array3d(quiet)[x : x + w, y : y + h]
        rows_with_wave = lambda area: np.flatnonzero(  # noqa: E731
            (area == np.array(SCOPE_COLOUR)).all(axis=2).any(axis=0)
        )
        assert len(rows_with_wave(loud_rows)) > h * 0.6
        assert len(rows_with_wave(flat_rows)) <= 2
        assert abs(rows_with_wave(flat_rows).mean() - h / 2) < 2

    def test_positive_samples_are_drawn_above_the_middle_line(self):
        viewer, up = self.draw(sp.Float(0.5), at=2.0)
        _, down = self.draw(sp.Float(-0.5), at=2.0)
        x, y, w, h = viewer.layout.scope
        for shot, above in ((up, True), (down, False)):
            area = pygame.surfarray.array3d(shot)[x : x + w, y : y + h]
            rows = np.flatnonzero(
                (area == np.array(SCOPE_COLOUR)).all(axis=2).any(axis=0)
            )
            assert (rows.mean() < h / 2) == above

    def test_the_scope_shows_the_right_number_of_cycles(self):
        # 40 ms of a 200 Hz tone is eight cycles
        viewer, surface = self.draw(sine(200), at=2.0)
        x, y, w, h = viewer.layout.scope
        area = pygame.surfarray.array3d(surface)[x : x + w, y : y + h]
        trace = (area == np.array(SCOPE_COLOUR)).all(axis=2)
        middle = h // 2
        centre_line = np.array(
            [np.flatnonzero(trace[column]).mean() for column in range(w)]
        )
        side = np.sign(np.where(abs(centre_line - middle) < 4, 0, centre_line - middle))
        side = side[side != 0]  # ignore the wobble right at the middle
        crossings = np.sum(np.diff(side) != 0)
        assert crossings == pytest.approx(16, abs=3)

    def test_the_scope_is_blank_before_anything_has_played(self):
        viewer, surface = self.draw(at=0.0)
        x, y, w, h = viewer.layout.scope
        area = pygame.surfarray.array3d(surface)[x : x + w, y : y + h]
        trace = (area == np.array(SCOPE_COLOUR)).all(axis=2)
        assert np.flatnonzero(trace.any(axis=1)).size == w  # a line, drawn flat
        assert np.flatnonzero(trace.any(axis=0)).size <= 2

    def test_the_spectrogram_lights_up_the_row_of_the_tone(self):
        viewer, surface = self.draw(sine(1000), at=2.0)
        x, y, w, h = viewer.layout.spectrogram
        pixels = pygame.surfarray.array3d(surface)
        column = pixels[x + w - 40, y : y + h].astype(int).sum(axis=1)
        brightest = h - 1 - int(np.argmax(column))  # counted up from the bottom
        expected = (
            row_of_frequency(1000, SPECTROGRAM_ROWS, RATE)
            / (SPECTROGRAM_ROWS - 1)
            * (h - 1)
        )
        assert brightest == pytest.approx(expected, abs=6)

    def test_the_spectrogram_scrolls_with_the_playhead(self):
        # Silence, then a tone at 2 s: it is on screen only once the playhead passes

        beep = sine(1000) * sp.Piecewise((0.0, TIME < 2.0), (1.0, True))
        before, shot_before = self.draw(beep, at=1.0)
        after, shot_after = self.draw(beep, at=2.5)
        x, y, w, h = before.layout.spectrogram

        def bright(shot):
            area = pygame.surfarray.array3d(shot)
            return area[x + 120 : x + w - 3, y + 20 : y + h].max()

        assert bright(shot_before) < 40
        assert bright(shot_after) > 200

    def test_the_newest_column_is_at_the_right_edge(self):

        beep = sine(1000) * sp.Piecewise((0.0, TIME < 2.0), (1.0, True))
        viewer, surface = self.draw(beep, at=2.0 + 8 * HOP / RATE)
        x, y, w, h = viewer.layout.spectrogram
        area = pygame.surfarray.array3d(surface)[
            x + 120 : x + w - 3, y + 20 : y + h
        ].astype(int)
        lit = np.flatnonzero(area.sum(axis=2).max(axis=1) > 300) + 120
        # The tone began about eight columns ago, so it fills the right edge
        # of the picture and no more than that
        assert lit.min() > w - 3 - 14 * PIXELS_PER_COLUMN
        assert lit.max() >= w - 8

    def test_the_spectrogram_has_a_playhead_at_its_right_edge(self):
        viewer, surface = self.draw()
        x, y, w, h = viewer.layout.spectrogram
        edge = pygame.surfarray.array3d(surface)[x + w - 1, y : y + h]
        assert (edge == np.array(PLAYHEAD)).all()

    def test_the_play_button_changes_with_the_state(self):
        paused, shot_paused = self.draw(playing=False)
        playing, shot_playing = self.draw(playing=True)
        x, y, w, h = paused.layout.play
        a = pygame.surfarray.array3d(shot_paused)[x : x + w, y : y + h]
        b = pygame.surfarray.array3d(shot_playing)[x : x + w, y : y + h]
        assert not np.array_equal(a, b)

    def test_the_time_readout_changes_with_the_time(self):
        a, shot_a = self.draw(at=0.0)
        b, shot_b = self.draw(at=3.0)
        x, y, w, h = a.layout.header
        area = (x, y, w, 20)
        left = pygame.surfarray.array3d(shot_a)[area[0] : area[0] + w, y : y + 20]
        right = pygame.surfarray.array3d(shot_b)[area[0] : area[0] + w, y : y + 20]
        assert not np.array_equal(left, right)

    def test_a_quiet_signal_draws_a_quiet_spectrogram(self):
        loud, shot_loud = self.draw(sine(1000))
        quiet, shot_quiet = self.draw(sine(1000) * 0.001)
        x, y, w, h = loud.layout.spectrogram

        def brightness(shot):
            area = pygame.surfarray.array3d(shot)
            return area[x : x + w - 3, y : y + h].mean()

        assert brightness(shot_quiet) < brightness(shot_loud) / 2

    def test_drawing_does_not_change_the_playback(self):
        viewer, _ = self.draw(playing=True)
        assert viewer.player.position == int(2.0 * RATE)

    def test_the_overview_is_worked_out_once_per_width(self):
        viewer, surface = self.draw()
        viewer.draw(surface)
        assert len(viewer._overview) == 1
        viewer.draw(pygame.Surface((800, 600)))
        assert len(viewer._overview) == 2


class TestSnapshot:
    def test_it_writes_a_png_of_the_window_without_a_display(self, tmp_path):
        out = tmp_path / "deep" / "dir" / "shot.png"
        snapshot(sine(440), 2.0, 1.0, out, sample_rate=RATE, size=(640, 400))
        image = pygame.image.load(str(out))
        assert image.get_size() == (640, 400)
        assert pygame.surfarray.array3d(image).std() > 5

    def test_the_time_chosen_shows_in_the_picture(self, tmp_path):
        snapshot(sine(440), 4.0, 0.5, tmp_path / "a.png", RATE)
        snapshot(sine(440), 4.0, 3.0, tmp_path / "b.png", RATE)
        a = pygame.surfarray.array3d(pygame.image.load(str(tmp_path / "a.png")))
        b = pygame.surfarray.array3d(pygame.image.load(str(tmp_path / "b.png")))
        assert not np.array_equal(a, b)

    def test_the_picture_matches_what_the_viewer_draws(self, tmp_path):
        snapshot(
            sine(440), 2.0, 1.0, tmp_path / "s.png", RATE, title="x", size=(500, 400)
        )
        viewer = _Viewer(make_player(sine(440), 2.0), "x")
        viewer.player.seek(1.0)
        surface = pygame.Surface((500, 400))
        viewer.draw(surface)
        saved = pygame.surfarray.array3d(pygame.image.load(str(tmp_path / "s.png")))
        np.testing.assert_array_equal(saved, pygame.surfarray.array3d(surface))

    def test_it_can_show_the_pause_button_or_the_pressed_one(self, tmp_path):
        snapshot(sine(440), 2.0, 1.0, tmp_path / "a.png", RATE, playing=False)
        snapshot(sine(440), 2.0, 1.0, tmp_path / "b.png", RATE, playing=True)
        a = pygame.image.load(str(tmp_path / "a.png"))
        b = pygame.image.load(str(tmp_path / "b.png"))
        assert not np.array_equal(
            pygame.surfarray.array3d(a), pygame.surfarray.array3d(b)
        )

    def test_a_filtered_signal_works_because_it_is_compiled_whole(self, tmp_path):
        from symsound.filters import low_pass_filter

        snapshot(low_pass_filter(sine(2000), 300), 2.0, 1.0, tmp_path / "f.png", RATE)
        assert (tmp_path / "f.png").exists()

    def test_a_non_positive_duration_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="duration"):
            snapshot(sine(440), 0, 0, tmp_path / "x.png")


class FakeScreenViewer(_Viewer):
    """Quits after a few frames, and counts them"""

    frames_before_quit = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drawn = 0
        self.sizes = []

    def draw(self, surface):
        super().draw(surface)
        self.drawn += 1
        self.sizes.append(surface.get_size())
        if self.drawn >= self.frames_before_quit:
            self.quit = True


class TestRun:
    def test_it_draws_frames_until_the_viewer_quits_then_closes_pygame(self):
        viewer = FakeScreenViewer(make_player())
        _run(viewer, size=(480, 320), fps=1000)
        assert viewer.drawn == 3
        assert viewer.sizes[0] == (480, 320)
        assert not pygame.get_init()

    def test_it_handles_the_events_that_arrive(self):
        viewer = FakeScreenViewer(make_player())
        viewer.frames_before_quit = 50
        original = viewer.handle
        seen = []

        def handle(event):
            seen.append(event.type)
            original(event)

        viewer.handle = handle
        # The first frame creates the window; a key sent before then is lost
        draw = viewer.draw

        def draw_and_press(surface):
            draw(surface)
            if viewer.drawn == 1:
                pygame.event.post(key(pygame.K_q))

        viewer.draw = draw_and_press
        _run(viewer, size=(300, 260), fps=1000)
        assert pygame.KEYDOWN in seen
        assert viewer.quit
        assert viewer.drawn < 50

    def test_the_playhead_runs_on_the_clock_when_there_is_no_audio(self):
        player = make_player()
        viewer = FakeScreenViewer(player, drive_clock=True)
        viewer.frames_before_quit = 5
        player.play()
        _run(viewer, size=(300, 260), fps=200)
        assert player.position > 0

    def test_pygame_is_closed_even_if_drawing_fails(self):
        class Broken(FakeScreenViewer):
            def draw(self, surface):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            _run(Broken(make_player()), size=(300, 260))
        assert not pygame.get_init()


class TestVisualize:
    @pytest.fixture
    def opened(self, monkeypatch):
        """Replace the window loop, and keep the viewer it was asked to run"""
        seen = {}

        def fake_run(viewer, size):
            seen["viewer"] = viewer
            seen["size"] = size

        monkeypatch.setattr(visualizer, "_run", fake_run)
        return seen

    def test_it_plays_the_compiled_signal_through_the_speakers_and_stops_after(
        self, fake_sounddevice, opened
    ):
        visualize(sine(440), 1.0, RATE, title="tone")
        (stream,) = fake_sounddevice.streams
        assert stream.events == ["start", "stop", "close"]
        assert stream.options["samplerate"] == RATE
        viewer = opened["viewer"]
        assert viewer.title == "tone"
        assert viewer.player.playing
        assert not viewer.drive_clock
        np.testing.assert_allclose(
            viewer.player.samples, compile(sine(440), 1.0, RATE), atol=1e-9
        )

    def test_the_speakers_pull_from_the_viewers_player(self, fake_sounddevice, opened):
        visualize(sine(440), 1.0, RATE)
        outdata = np.zeros((256, 1), dtype=np.float32)
        fake_sounddevice.streams[0].callback(outdata, 256, None, None)
        player = opened["viewer"].player
        assert player.position == 256
        np.testing.assert_allclose(
            outdata[:, 0], player.samples[:256].astype(np.float32), atol=1e-6
        )

    def test_silent_makes_no_sound_and_runs_the_picture_on_the_clock(
        self, fake_sounddevice, opened
    ):
        visualize(sine(440), 1.0, RATE, silent=True)
        assert fake_sounddevice.streams == []
        assert opened["viewer"].drive_clock

    def test_it_can_start_paused(self, fake_sounddevice, opened):
        visualize(sine(440), 1.0, RATE, autoplay=False)
        assert not opened["viewer"].player.playing

    def test_the_window_size_is_passed_on(self, fake_sounddevice, opened):
        visualize(sine(440), 1.0, RATE, size=(700, 500))
        assert opened["size"] == (700, 500)

    def test_the_stream_is_stopped_even_if_the_window_fails(
        self, fake_sounddevice, monkeypatch
    ):
        def broken(viewer, size):
            raise RuntimeError("boom")

        monkeypatch.setattr(visualizer, "_run", broken)
        with pytest.raises(RuntimeError):
            visualize(sine(440), 1.0, RATE)
        assert fake_sounddevice.streams[0].events == ["start", "stop", "close"]

    def test_a_non_positive_duration_is_refused_before_anything_opens(
        self, fake_sounddevice, opened
    ):
        with pytest.raises(ValueError, match="duration"):
            visualize(sine(440), 0.0)
        assert fake_sounddevice.streams == [] and not opened

    def test_the_default_duration_is_ten_seconds(self, fake_sounddevice, opened):
        visualize(sine(440), sample_rate=RATE)
        assert opened["viewer"].player.duration == pytest.approx(10.0)

    def test_the_whole_thing_runs_in_a_real_pygame_window(
        self, fake_sounddevice, monkeypatch
    ):
        # No fake window loop: a real one on the dummy driver. The "user" presses
        # q from inside the third frame, so there is no timing to get wrong
        frames = []
        original = _Viewer.advance

        def advance(self, seconds):
            original(self, seconds)
            frames.append(seconds)
            if len(frames) == 3:
                self.handle(key(pygame.K_q))

        monkeypatch.setattr(_Viewer, "advance", advance)
        visualize(sine(440), 1.0, RATE, silent=True, size=(400, 300))
        assert len(frames) == 3
        assert not pygame.get_init()


class TestOptionalDependency:
    def test_importing_the_visualizer_without_pygame_explains_what_to_install(
        self, monkeypatch
    ):
        import importlib

        from symsound import _draw

        monkeypatch.setitem(sys.modules, "pygame", None)
        with pytest.raises(ImportError, match=r"symsound\[gui\]"):
            importlib.reload(_draw)

    def test_the_rest_of_the_package_never_imports_pygame(self):
        code = (
            "import sys\n"
            "import symsound.live, symsound.analysis\n"
            "import symsound.waves.utils, symsound.filters\n"
            "print('pygame' in sys.modules)\n"
        )
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
        assert result.stdout.strip() == "False"

    def test_the_library_does_not_print_pygames_banner(self):
        code = "import symsound.visualizer\n"
        src = str(Path(symsound.__file__).resolve().parents[1])
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH")]))
        env.pop("PYGAME_HIDE_SUPPORT_PROMPT", None)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        assert result.stdout == ""
