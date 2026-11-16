"""Drawing shared by the windows (`visualize` and `keyboard`).

Private: the names here are not part of the library's API. The scope and the
spectrogram are drawn the same way whether they show a finished signal or what
is being played right now, so that code lives in one place.
"""

import os
from typing import Protocol

import matplotlib
import numpy as np
import numpy.typing as npt

from symsound.analysis import row_of_frequency

# pygame announces itself on import: not wanted in a library, nor in the docs
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
try:
    import pygame
except ImportError as error:
    raise ImportError(
        "the visualizer needs pygame: install it with pip install 'symsound[gui]'"
    ) from error

Rect = tuple[int, int, int, int]  # x, y, width, height
Pixels = npt.NDArray[np.uint8]

WINDOW_SIZE = (1000, 640)
SCOPE_SECONDS = 0.04  # how much of the signal the scope shows
SPECTROGRAM_ROWS = 256
PIXELS_PER_COLUMN = 2  # on screen, so about 5 s of history across 1000 px
MARGIN = 12
BAR_HEIGHT = 34
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


def _colourise(levels: npt.ArrayLike) -> Pixels:
    """Spectrogram levels from 0 to 1 as RGB pixels (appends a size-3 axis)"""
    clipped = np.clip(np.asarray(levels, dtype=float), 0.0, 1.0)
    return _PALETTE[np.round(clipped * 255).astype(int)]


def _inside(rect: Rect, x: float, y: float) -> bool:
    left, top, width, height = rect
    return left <= x < left + width and top <= y < top + height


class _Panels:
    """Fonts, and the pieces every window is made of"""

    def __init__(self) -> None:
        pygame.font.init()
        self.font = pygame.font.Font(None, 22)
        self.small = pygame.font.Font(None, 17)

    def text(
        self,
        surface: pygame.Surface,
        text: str,
        font: pygame.font.Font,
        colour: tuple[int, int, int],
        x: int,
        y: int,
        *,
        right: bool = False,
        centre: Rect | None = None,
    ) -> None:
        image = font.render(text, True, colour)
        if centre is not None:
            left, top, width, height = centre
            x = left + (width - image.get_width()) // 2
            y = top + (height - image.get_height()) // 2
        elif right:
            x -= image.get_width()
        surface.blit(image, (x, y))

    def button(self, surface: pygame.Surface, rect: Rect, label: str) -> None:
        pygame.draw.rect(surface, BUTTON, rect, border_radius=6)
        self.text(surface, label, self.font, TEXT, 0, 0, centre=rect)

    def panel(self, surface: pygame.Surface, rect: Rect, caption: str) -> None:
        pygame.draw.rect(surface, PANEL, rect)
        self.text(surface, caption, self.small, MUTED, rect[0] + 6, rect[1] + 4)

    def scope(
        self,
        surface: pygame.Surface,
        rect: Rect,
        recent: npt.ArrayLike,
        sample_rate: int,
    ) -> None:
        """The latest samples as a wave. Fewer than the scope shows are drawn
        at its right-hand end, with silence before them."""
        x, y, width, height = rect
        self.panel(surface, rect, f"scope  last {SCOPE_SECONDS * 1000:.0f} ms")
        middle = y + height / 2
        for level in (-1.0, -0.5, 0.0, 0.5, 1.0):
            line = middle - level * (height / 2 - 4)
            pygame.draw.line(surface, GRID, (x, line), (x + width - 1, line))
        count = round(SCOPE_SECONDS * sample_rate)
        latest = np.asarray(recent, dtype=float)[-count:]
        shown = np.zeros(count)
        shown[count - len(latest) :] = latest
        picks = np.linspace(0, count - 1, width).astype(int)
        heights = np.clip(shown[picks], -1.0, 1.0) * (height / 2 - 4)
        points = [(x + i, middle - h) for i, h in enumerate(heights)]
        pygame.draw.lines(surface, SCOPE_COLOUR, False, points, 1)

    def spectrogram(
        self,
        surface: pygame.Surface,
        rect: Rect,
        levels: npt.ArrayLike,
        sample_rate: int,
    ) -> None:
        """Columns of levels (newest last, shape (columns, rows)) as a picture
        that ends at the playhead on the right. Fewer columns than fit are
        drawn at the right-hand end."""
        x, y, width, height = rect
        pygame.draw.rect(surface, PANEL, rect)
        fit = max(width // PIXELS_PER_COLUMN, 1)
        have = np.asarray(levels, dtype=float).reshape(-1, SPECTROGRAM_ROWS)[-fit:]
        picture = np.zeros((fit, SPECTROGRAM_ROWS, 3), dtype=np.uint8)
        picture[fit - len(have) :] = _colourise(have)
        # surfaces are (x, y) and y grows downwards, so flip the rows
        image = pygame.surfarray.make_surface(picture[:, ::-1, :])
        scaled = pygame.transform.scale(image, (fit * PIXELS_PER_COLUMN, height))
        surface.blit(scaled, (x, y))
        for frequency in FREQUENCY_LABELS:
            if frequency >= sample_rate / 2:
                continue
            row = row_of_frequency(frequency, SPECTROGRAM_ROWS, sample_rate)
            line = y + height - 1 - row / (SPECTROGRAM_ROWS - 1) * (height - 1)
            pygame.draw.line(surface, GRID, (x, line), (x + 8, line))
            name = (
                f"{frequency // 1000} kHz" if frequency >= 1000 else f"{frequency} Hz"
            )
            self.text(surface, name, self.small, TEXT, x + 12, int(line) - 6)
        self.text(
            surface, "spectrogram", self.small, TEXT, x + width - 6, y + 4, right=True
        )
        pygame.draw.line(
            surface, PLAYHEAD, (x + width - 1, y), (x + width - 1, y + height)
        )


def _run(viewer: _Window, size: tuple[int, int] = WINDOW_SIZE, fps: int = 60) -> None:
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


class _Window(Protocol):
    """What `_run` needs of a window: the same shape for every tool"""

    title: str
    quit: bool

    def handle(self, event: pygame.event.Event) -> None: ...

    def draw(self, surface: pygame.Surface) -> None: ...

    def advance(self, seconds: float) -> None: ...
