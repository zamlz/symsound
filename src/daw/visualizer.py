# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""A window that shows a signal as it plays: a scope, a spectrogram, a seek bar.

`visualize(expression, duration)` compiles the signal, plays it, and draws what
is being heard. Space pauses, R replays, the arrow keys step through it, and
you can click or drag on the seek bar. Everything it draws comes from
`daw.analysis`; this module only turns those numbers into pixels.

It needs pygame (`pip install 'daw[gui]'`), which the rest of the package does not.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sympy.core.expr import Expr

from daw._draw import (
    BACKGROUND,
    BAR_HEIGHT,
    MARGIN,
    MUTED,
    NOT_PLAYED,
    PANEL,
    PLAYED,
    PLAYHEAD,
    SPECTROGRAM_ROWS,
    TEXT,
    WINDOW_SIZE,
    Rect,
    _inside,
    _Panels,
    _run,
    pygame,
)
from daw.analysis import HOP, peak_envelope, spectrogram_columns
from daw.constants import DEFAULT_SAMPLE_RATE
from daw.live import LiveOutput, Player
from daw.waves.utils import compile

SEEK_HEIGHT = 64
SMALL_STEP = 1.0  # seconds jumped by an arrow key
BIG_STEP = 5.0  # ... with shift held


def _format_time(seconds: float) -> str:
    """Minutes, seconds and tenths: 83.25 -> '1:23.2'"""
    tenths = int(max(seconds, 0.0) * 10)
    minutes, rest = divmod(tenths, 600)
    return f"{minutes}:{rest // 10:02d}.{rest % 10}"


@dataclass(frozen=True)
class _Layout:
    """Where everything goes in a window of a given size"""

    play: Rect
    replay: Rect
    header: Rect
    scope: Rect
    spectrogram: Rect
    seek: Rect


def _layout_for(width: int, height: int) -> _Layout:
    """Split a window into controls, scope, spectrogram and seek bar"""
    width, height = max(width, 320), max(height, 260)
    inner = width - 2 * MARGIN
    play = (MARGIN, MARGIN, 92, BAR_HEIGHT)
    replay = (MARGIN + 92 + 8, MARGIN, 92, BAR_HEIGHT)
    header_x = replay[0] + replay[2] + 12
    header = (header_x, MARGIN, width - MARGIN - header_x, BAR_HEIGHT)
    top = MARGIN + BAR_HEIGHT + MARGIN
    seek_y = height - MARGIN - SEEK_HEIGHT
    room = seek_y - MARGIN - top
    scope_height = max(room * 3 // 10, 40)
    spectrogram_y = top + scope_height + MARGIN
    return _Layout(
        play=play,
        replay=replay,
        header=header,
        scope=(MARGIN, top, inner, scope_height),
        spectrogram=(
            MARGIN,
            spectrogram_y,
            inner,
            max(seek_y - MARGIN - spectrogram_y, 20),
        ),
        seek=(MARGIN, seek_y, inner, SEEK_HEIGHT),
    )


def _seek_fraction(rect: Rect, x: float) -> float:
    """How far along the seek bar an x position is, from 0 to 1"""
    left, _, width, _ = rect
    return min(max((x - left) / width, 0.0), 1.0)


class _Viewer:
    """The picture of a `Player`: handles input and draws onto any surface.

    It never opens a window itself (`visualize` does that), so it can be
    drawn onto an off-screen surface, which is how it is tested."""

    def __init__(
        self, player: Player, title: str = "daw", *, drive_clock: bool = False
    ):
        self.player = player
        self.title = title
        # With no audio to move the playhead, the display has to
        self.drive_clock = drive_clock
        self.quit = False
        self.layout = _layout_for(*WINDOW_SIZE)
        self._dragging = False
        self._resume = False
        self._panels = _Panels()
        rate = player.sample_rate
        self._spectrogram = spectrogram_columns(player.samples, rate, SPECTROGRAM_ROWS)
        self._overview: dict[int, tuple[Any, Any]] = {}

    # input

    def handle(self, event: pygame.event.Event) -> None:
        player = self.player
        if event.type == pygame.QUIT:
            self.quit = True
        elif event.type == pygame.VIDEORESIZE:
            self.layout = _layout_for(event.w, event.h)
        elif event.type == pygame.KEYDOWN:
            self._key(event.key, event.mod)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            x, y = event.pos
            if _inside(self.layout.play, x, y):
                player.toggle()
            elif _inside(self.layout.replay, x, y):
                player.replay()
            elif _inside(self.layout.seek, x, y):
                self._dragging = True
                self._resume = player.playing
                player.pause()
                player.seek(_seek_fraction(self.layout.seek, x) * player.duration)
        elif event.type == pygame.MOUSEMOTION and self._dragging:
            x = event.pos[0]
            player.seek(_seek_fraction(self.layout.seek, x) * player.duration)
        elif (
            event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self._dragging
        ):
            self._dragging = False
            if self._resume:
                player.play()

    def _key(self, key: int, modifiers: int) -> None:
        player = self.player
        step = BIG_STEP if modifiers & pygame.KMOD_SHIFT else SMALL_STEP
        if key in (pygame.K_ESCAPE, pygame.K_q):
            self.quit = True
        elif key == pygame.K_SPACE:
            player.toggle()
        elif key == pygame.K_r:
            player.replay()
        elif key == pygame.K_HOME:
            player.seek(0.0)
        elif key == pygame.K_LEFT:
            player.seek(player.seconds - step)
        elif key == pygame.K_RIGHT:
            player.seek(player.seconds + step)

    def advance(self, seconds: float) -> None:
        """Move the playhead on by wall-clock time, when there is no audio to"""
        if self.drive_clock:
            self.player.render(round(seconds * self.player.sample_rate))

    # drawing

    def draw(self, surface: pygame.Surface) -> None:
        self.layout = _layout_for(*surface.get_size())
        surface.fill(BACKGROUND)
        self._draw_controls(surface)
        player = self.player
        stop = player.position
        self._panels.scope(
            surface, self.layout.scope, player.samples[:stop], player.sample_rate
        )
        newest = min(stop // HOP, len(self._spectrogram) - 1)
        self._panels.spectrogram(
            surface,
            self.layout.spectrogram,
            self._spectrogram[: newest + 1],
            player.sample_rate,
        )
        self._draw_seek(surface)

    def _draw_controls(self, surface: pygame.Surface) -> None:
        panels = self._panels
        panels.button(
            surface, self.layout.play, "Pause" if self.player.playing else "Play"
        )
        panels.button(surface, self.layout.replay, "Replay")
        x, y, width, height = self.layout.header
        player = self.player
        clock = f"{_format_time(player.seconds)} / {_format_time(player.duration)}"
        panels.text(surface, self.title, panels.font, TEXT, x, y + 2)
        panels.text(surface, clock, panels.font, MUTED, x + width, y + 2, right=True)
        hint = "space play/pause   r replay   arrows seek (shift: 5 s)   q quit"
        panels.text(surface, hint, panels.small, MUTED, x, y + 21)

    def _overview_for(self, width: int) -> tuple[Any, Any]:
        if width not in self._overview:
            self._overview[width] = peak_envelope(self.player.samples, width)
        return self._overview[width]

    def _draw_seek(self, surface: pygame.Surface) -> None:
        rect = self.layout.seek
        x, y, width, height = rect
        pygame.draw.rect(surface, PANEL, rect)
        low, high = self._overview_for(width)
        middle = y + height / 2
        reach = height / 2 - 4
        now = round(width * self.player.position / len(self.player.samples))
        for column in range(width):
            colour = PLAYED if column < now else NOT_PLAYED
            top = middle - float(np.clip(high[column], -1, 1)) * reach
            bottom = middle - float(np.clip(low[column], -1, 1)) * reach
            pygame.draw.line(
                surface, colour, (x + column, top), (x + column, max(bottom, top + 1))
            )
        line = x + min(now, width - 1)
        pygame.draw.line(surface, PLAYHEAD, (line, y), (line, y + height - 1), 2)




def visualize(
    expression: Expr,
    duration: float = 10.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    title: str = "daw",
    silent: bool = False,
    autoplay: bool = True,
    size: tuple[int, int] = WINDOW_SIZE,
) -> None:
    """Play a signal in a window that shows it: scope, spectrogram and seek bar.

    The whole signal is compiled first, so everything works, filters included.
    With `silent` no sound is made and the picture runs on the clock instead.
    With `autoplay` off it starts paused. Returns when the window is closed."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    player = Player(compile(expression, duration, sample_rate), sample_rate)
    viewer = _Viewer(player, title, drive_clock=silent)
    if autoplay:
        player.play()
    output = None if silent else LiveOutput(player, sample_rate)
    if output is not None:
        output.start()
    try:
        _run(viewer, size)
    finally:
        if output is not None:
            output.stop()


def snapshot(
    expression: Expr,
    duration: float,
    at: float,
    output_file: str | os.PathLike[str],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    title: str = "daw",
    playing: bool = False,
    size: tuple[int, int] = WINDOW_SIZE,
) -> None:
    """Save a picture of the window as it would look `at` seconds in.

    Nothing is opened or played: it draws on an off-screen image, so it works
    without a display (the docs use it for their pictures)."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    player = Player(compile(expression, duration, sample_rate), sample_rate)
    player.seek(at)
    if playing:
        player.play()
    viewer = _Viewer(player, title)
    surface = pygame.Surface(size)
    viewer.draw(surface)
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(path))
