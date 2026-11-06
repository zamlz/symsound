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

import matplotlib
import numpy as np
import numpy.typing as npt
from sympy.core.expr import Expr

from daw.analysis import (
    HOP,
    peak_envelope,
    row_of_frequency,
    spectrogram_columns,
)
from daw.constants import DEFAULT_SAMPLE_RATE
from daw.live import LiveOutput, Player
from daw.waves.utils import compile

# pygame announces itself on import: not wanted in a library, nor in the docs
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
try:
    import pygame
except ImportError as error:
    raise ImportError(
        "the visualizer needs pygame: install it with pip install 'daw[gui]'"
    ) from error

Rect = tuple[int, int, int, int]  # x, y, width, height

WINDOW_SIZE = (1000, 640)
SCOPE_SECONDS = 0.04  # how much of the signal the scope shows
SPECTROGRAM_ROWS = 256
PIXELS_PER_COLUMN = 2  # on screen, so about 5 s of history across 1000 px
MARGIN = 12
BAR_HEIGHT = 34
SEEK_HEIGHT = 64
SMALL_STEP = 1.0  # seconds jumped by an arrow key
BIG_STEP = 5.0  # ... with shift held
FREQUENCY_LABELS = (100, 1000, 10000)

BACKGROUND = (16, 18, 24)
PANEL = (24, 27, 36)
GRID = (50, 56, 70)
TEXT = (225, 228, 236)
MUTED = (140, 148, 166)
BUTTON = (44, 50, 66)
SCOPE_COLOUR = (110, 200, 255)
PLAYHEAD = (255, 176, 64)
PLAYED = (110, 200, 255)
NOT_PLAYED = (70, 84, 104)

# 256 colours from black to bright yellow, one per spectrogram level
_PALETTE = (matplotlib.colormaps["magma"](np.linspace(0, 1, 256))[:, :3] * 255).astype(
    np.uint8
)

Pixels = npt.NDArray[np.uint8]


def _colourise(levels: npt.ArrayLike) -> Pixels:
    """Spectrogram levels from 0 to 1 as RGB pixels (appends a size-3 axis)"""
    clipped = np.clip(np.asarray(levels, dtype=float), 0.0, 1.0)
    return _PALETTE[np.round(clipped * 255).astype(int)]


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


def _inside(rect: Rect, x: float, y: float) -> bool:
    left, top, width, height = rect
    return left <= x < left + width and top <= y < top + height


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
        pygame.font.init()
        self.player = player
        self.title = title
        # With no audio to move the playhead, the display has to
        self.drive_clock = drive_clock
        self.quit = False
        self.layout = _layout_for(*WINDOW_SIZE)
        self._dragging = False
        self._resume = False
        self._font = pygame.font.Font(None, 22)
        self._small = pygame.font.Font(None, 17)
        rate = player.sample_rate
        grid = spectrogram_columns(player.samples, rate, SPECTROGRAM_ROWS)
        self._spectrogram = _colourise(grid)  # (columns, rows, 3)
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
        self._draw_scope(surface)
        self._draw_spectrogram(surface)
        self._draw_seek(surface)

    def _text(self, surface, text, font, colour, x, y, *, right=False, centre=None):
        image = font.render(text, True, colour)
        if centre is not None:
            left, top, width, height = centre
            x = left + (width - image.get_width()) // 2
            y = top + (height - image.get_height()) // 2
        elif right:
            x -= image.get_width()
        surface.blit(image, (x, y))

    def _draw_controls(self, surface: pygame.Surface) -> None:
        for rect, label in (
            (self.layout.play, "Pause" if self.player.playing else "Play"),
            (self.layout.replay, "Replay"),
        ):
            pygame.draw.rect(surface, BUTTON, rect, border_radius=6)
            self._text(surface, label, self._font, TEXT, 0, 0, centre=rect)
        x, y, width, height = self.layout.header
        player = self.player
        clock = f"{_format_time(player.seconds)} / {_format_time(player.duration)}"
        self._text(surface, self.title, self._font, TEXT, x, y + 2)
        self._text(surface, clock, self._font, MUTED, x + width, y + 2, right=True)
        hint = "space play/pause   r replay   arrows seek (shift: 5 s)   q quit"
        self._text(surface, hint, self._small, MUTED, x, y + 21)

    def _panel(self, surface, rect: Rect, caption: str) -> None:
        pygame.draw.rect(surface, PANEL, rect)
        self._text(surface, caption, self._small, MUTED, rect[0] + 6, rect[1] + 4)

    def _draw_scope(self, surface: pygame.Surface) -> None:
        rect = self.layout.scope
        x, y, width, height = rect
        self._panel(surface, rect, f"scope  last {SCOPE_SECONDS * 1000:.0f} ms")
        middle = y + height / 2
        for level in (-1.0, -0.5, 0.0, 0.5, 1.0):
            line = middle - level * (height / 2 - 4)
            pygame.draw.line(surface, GRID, (x, line), (x + width - 1, line))
        player = self.player
        count = round(SCOPE_SECONDS * player.sample_rate)
        stop = player.position
        recent = player.samples[max(stop - count, 0) : stop]
        shown = np.zeros(count)
        shown[count - len(recent) :] = recent
        picks = np.linspace(0, count - 1, width).astype(int)
        heights = np.clip(shown[picks], -1.0, 1.0) * (height / 2 - 4)
        points = [(x + i, middle - h) for i, h in enumerate(heights)]
        pygame.draw.lines(surface, SCOPE_COLOUR, False, points, 1)

    def _draw_spectrogram(self, surface: pygame.Surface) -> None:
        rect = self.layout.spectrogram
        x, y, width, height = rect
        pygame.draw.rect(surface, PANEL, rect)
        columns = max(width // PIXELS_PER_COLUMN, 1)
        newest = min(self.player.position // HOP, len(self._spectrogram) - 1)
        first = newest - columns + 1
        window = np.zeros((columns, SPECTROGRAM_ROWS, 3), dtype=np.uint8)
        have = self._spectrogram[max(first, 0) : newest + 1]
        window[columns - len(have) :] = have
        # surfaces are (x, y) and y grows downwards, so flip the rows
        image = pygame.surfarray.make_surface(window[:, ::-1, :])
        surface.blit(
            pygame.transform.scale(image, (columns * PIXELS_PER_COLUMN, height)), (x, y)
        )
        rate = self.player.sample_rate
        for frequency in FREQUENCY_LABELS:
            if frequency >= rate / 2:
                continue
            row = row_of_frequency(frequency, SPECTROGRAM_ROWS, rate)
            line = y + height - 1 - row / (SPECTROGRAM_ROWS - 1) * (height - 1)
            pygame.draw.line(surface, GRID, (x, line), (x + 8, line))
            name = (
                f"{frequency // 1000} kHz" if frequency >= 1000 else f"{frequency} Hz"
            )
            self._text(surface, name, self._small, TEXT, x + 12, int(line) - 6)
        self._text(
            surface, "spectrogram", self._small, TEXT, x + width - 6, y + 4, right=True
        )
        pygame.draw.line(
            surface, PLAYHEAD, (x + width - 1, y), (x + width - 1, y + height)
        )

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


def _run(viewer: _Viewer, size: tuple[int, int] = WINDOW_SIZE, fps: int = 60) -> None:
    """Open a window for the viewer and keep it going until it is closed"""
    pygame.init()
    try:
        screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption(viewer.title)
        clock = pygame.time.Clock()
        while not viewer.quit:
            for event in pygame.event.get():
                viewer.handle(event)
            viewer.draw(screen)
            pygame.display.flip()
            viewer.advance(clock.tick(fps) / 1000)
    finally:
        pygame.quit()


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
