# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""A keyboard you can play: the keys of your computer make the notes of a signal.

`keyboard(f)` takes a function from a `Note` to a signal, such as
`lambda note: sine(note) * adsr_envelope(...)`, and opens a window with a piano
on screen, a scope and a spectrogram. The letter keys play the notes (see
`KEYS`), the on-screen piano can be clicked, and the picture shows what is
sounding as it happens.

The slow, symbolic work of turning the function into something fast is done for
every note before the window opens (`daw.live.Instrument`), so playing is only
a little arithmetic.

It needs pygame (`pip install 'daw[gui]'`), which the rest of the package does not.
"""

import os
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from pathlib import Path

from sympy.core.expr import Expr

from daw._draw import (
    BACKGROUND,
    BAR_HEIGHT,
    MARGIN,
    MUTED,
    PANEL,
    PLAYHEAD,
    TEXT,
    WINDOW_SIZE,
    Rect,
    _inside,
    _Panels,
    _run,
    pygame,
)
from daw.analysis import LiveSpectrogram
from daw.constants import DEFAULT_SAMPLE_RATE, Note
from daw.live import Instrument, LiveOutput, Mixer, RingBuffer

# (pygame key, label, semitones above the C of the current octave): a piano
# laid out over the keyboard, white keys on the home row and black keys above
KEYS = [
    (pygame.K_a, "A", 0),
    (pygame.K_w, "W", 1),
    (pygame.K_s, "S", 2),
    (pygame.K_e, "E", 3),
    (pygame.K_d, "D", 4),
    (pygame.K_f, "F", 5),
    (pygame.K_t, "T", 6),
    (pygame.K_g, "G", 7),
    (pygame.K_y, "Y", 8),
    (pygame.K_h, "H", 9),
    (pygame.K_u, "U", 10),
    (pygame.K_j, "J", 11),
    (pygame.K_k, "K", 12),
    (pygame.K_o, "O", 13),
    (pygame.K_l, "L", 14),
    (pygame.K_p, "P", 15),
    (pygame.K_SEMICOLON, ";", 16),
]
SEMITONE_OF_KEY = {key: semitone for key, _, semitone in KEYS}
LABEL_OF_SEMITONE = {semitone: label for _, label, semitone in KEYS}
BLACK = {1, 3, 6, 8, 10, 13, 15}  # the semitones that are black keys
WHITE = [s for _, _, s in KEYS if s not in BLACK]
FIRST_OCTAVE, LAST_OCTAVE = 1, 7
PIANO_HEIGHT = 150
RING_SAMPLES = 16384  # what the display can look back on
NOTES = list(Note)  # lowest first, one per semitone

WHITE_KEY = (235, 238, 246)
BLACK_KEY = (28, 30, 38)
LIT_WHITE = (255, 190, 90)
LIT_BLACK = (214, 120, 24)

# The note names of the semitones, in the table's spelling
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_at(octave: int, semitone: int) -> Note | None:
    """The note `semitone` steps above the C of `octave`, or None if the table
    does not go that far"""
    index = NOTES.index(Note[f"C{octave}"]) + semitone
    return NOTES[index] if 0 <= index < len(NOTES) else None


def note_name(note: Note) -> str:
    """'A4', 'C#5': a note as a musician writes it"""
    letter, octave = note.name[0], note.name[-1]
    return f"{letter}{'#' if len(note.name) == 3 else ''}{octave}"


@dataclass(frozen=True)
class _PianoKey:
    semitone: int
    black: bool
    rect: Rect


def _piano_keys(rect: Rect) -> list[_PianoKey]:
    """The keys of the on-screen piano, white ones first so black ones draw over"""
    x, y, width, height = rect
    step = width / len(WHITE)
    keys = [
        _PianoKey(
            semitone,
            False,
            (
                round(x + i * step),
                y,
                round(x + (i + 1) * step) - round(x + i * step),
                height,
            ),
        )
        for i, semitone in enumerate(WHITE)
    ]
    black_width = round(step * 0.6)
    for semitone in sorted(BLACK):
        edge = WHITE.index(semitone - 1) + 1  # the line between its two neighbours
        left = round(x + edge * step - black_width / 2)
        keys.append(_PianoKey(semitone, True, (left, y, black_width, height * 3 // 5)))
    return keys


def _key_under(rect: Rect, x: float, y: float) -> int | None:
    """The semitone of the piano key at a point; black keys are on top"""
    for key in reversed(_piano_keys(rect)):
        if _inside(key.rect, x, y):
            return key.semitone
    return None


@dataclass(frozen=True)
class _Layout:
    header: Rect
    scope: Rect
    spectrogram: Rect
    piano: Rect


def _layout_for(width: int, height: int) -> _Layout:
    width, height = max(width, 400), max(height, 380)
    inner = width - 2 * MARGIN
    top = MARGIN + BAR_HEIGHT + MARGIN
    piano_y = height - MARGIN - PIANO_HEIGHT
    room = piano_y - MARGIN - top
    scope_height = max(room * 3 // 10, 40)
    spectrogram_y = top + scope_height + MARGIN
    return _Layout(
        header=(MARGIN, MARGIN, inner, BAR_HEIGHT),
        scope=(MARGIN, top, inner, scope_height),
        spectrogram=(
            MARGIN,
            spectrogram_y,
            inner,
            max(piano_y - MARGIN - spectrogram_y, 20),
        ),
        piano=(MARGIN, piano_y, inner, PIANO_HEIGHT),
    )


class _Keyboard:
    """The keyboard window: input, sound and picture. Like the visualizer's
    window it draws onto any surface, so it can be tested off-screen."""

    def __init__(
        self,
        instrument: Instrument,
        mixer: Mixer,
        ring: RingBuffer,
        *,
        octave: int = 4,
        title: str = "daw keyboard",
        drive_clock: bool = False,
    ) -> None:
        if not FIRST_OCTAVE <= octave <= LAST_OCTAVE:
            raise ValueError(f"octave must be from {FIRST_OCTAVE} to {LAST_OCTAVE}")
        self.instrument = instrument
        self.mixer = mixer
        self.ring = ring
        self.octave = octave
        self.title = title
        # With no audio callback to pull sound through, the display must
        self.drive_clock = drive_clock
        self.quit = False
        self.layout = _layout_for(*WINDOW_SIZE)
        # Which note each finger (a key, or the mouse) started, so the same note
        # is released even if the octave has changed since
        self.sounding: dict[Hashable, tuple[Note, int]] = {}
        self._panels = _Panels()
        self._spectrogram = LiveSpectrogram(instrument.sample_rate)

    # input

    def press(self, finger: Hashable, semitone: int) -> None:
        note = note_at(self.octave, semitone)
        if note is None or finger in self.sounding:
            return
        self.sounding[finger] = (note, semitone)
        self.mixer.note_on(finger, self.instrument.source(note))

    def lift(self, finger: Hashable) -> None:
        if self.sounding.pop(finger, None) is not None:
            self.mixer.note_off(finger)

    def lift_all(self) -> None:
        for finger in list(self.sounding):
            self.lift(finger)

    def handle(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.quit = True
        elif event.type == pygame.VIDEORESIZE:
            self.layout = _layout_for(event.w, event.h)
        elif event.type == pygame.WINDOWFOCUSLOST:
            self.lift_all()  # the key-up would go to another window
        elif event.type == pygame.KEYDOWN:
            self._key_down(event.key)
        elif event.type == pygame.KEYUP:
            self.lift(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            semitone = _key_under(self.layout.piano, *event.pos)
            if semitone is not None:
                self.press("mouse", semitone)
        elif event.type == pygame.MOUSEMOTION and "mouse" in self.sounding:
            semitone = _key_under(self.layout.piano, *event.pos)
            if semitone != self.sounding["mouse"][1]:
                self.lift("mouse")
                if semitone is not None:
                    self.press("mouse", semitone)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.lift("mouse")

    def _key_down(self, key: int) -> None:
        if key in (pygame.K_ESCAPE, pygame.K_q):
            self.quit = True
        elif key == pygame.K_z:
            self.octave = max(self.octave - 1, FIRST_OCTAVE)
        elif key == pygame.K_x:
            self.octave = min(self.octave + 1, LAST_OCTAVE)
        elif key in SEMITONE_OF_KEY:
            self.press(key, SEMITONE_OF_KEY[key])

    def advance(self, seconds: float) -> None:
        """Move the display on: without audio the sound is made here, and the
        spectrogram takes in whatever has been played since the last frame"""
        if self.drive_clock:
            rate = self.instrument.sample_rate
            self.ring.write(self.mixer.render(round(seconds * rate)))
        self._spectrogram.update(self.ring)

    # drawing

    def draw(self, surface: pygame.Surface) -> None:
        self.layout = _layout_for(*surface.get_size())
        surface.fill(BACKGROUND)
        panels = self._panels
        x, y, width, height = self.layout.header
        panels.text(surface, self.title, panels.font, TEXT, x, y + 2)
        held = " ".join(note_name(note) for note, _ in self.sounding.values())
        panels.text(surface, held, panels.font, PLAYHEAD, x + width, y + 2, right=True)
        hint = f"octave {self.octave}   z/x change octave   click the piano   q quit"
        panels.text(surface, hint, panels.small, MUTED, x, y + 21)
        rate = self.instrument.sample_rate
        count = round(0.04 * rate)
        panels.scope(surface, self.layout.scope, self.ring.latest(count), rate)
        panels.spectrogram(
            surface, self.layout.spectrogram, self._spectrogram.levels, rate
        )
        self._draw_piano(surface)

    def _draw_piano(self, surface: pygame.Surface) -> None:
        lit = {semitone for _, semitone in self.sounding.values()}
        pygame.draw.rect(surface, PANEL, self.layout.piano)
        for key in _piano_keys(self.layout.piano):
            if key.black:
                colour = LIT_BLACK if key.semitone in lit else BLACK_KEY
            else:
                colour = LIT_WHITE if key.semitone in lit else WHITE_KEY
            pygame.draw.rect(surface, colour, key.rect, border_radius=3)
            pygame.draw.rect(surface, PANEL, key.rect, width=1, border_radius=3)
            kx, ky, kw, kh = key.rect
            ink = TEXT if key.black else BLACK_KEY
            label = LABEL_OF_SEMITONE[key.semitone]
            self._panels.text(
                surface,
                label,
                self._panels.font,
                ink,
                0,
                0,
                centre=(kx, ky + kh - 30, kw, 20),
            )
            note = note_at(self.octave, key.semitone)
            if note is not None and not key.black:
                self._panels.text(
                    surface,
                    note_name(note),
                    self._panels.small,
                    MUTED if key.semitone not in lit else BLACK_KEY,
                    0,
                    0,
                    centre=(kx, ky + kh - 52, kw, 16),
                )


def _check_envelope(instrument: Instrument, notes: Iterable[Note] = NOTES) -> None:
    """Refuse a signal that never learns when the key came up"""
    for note in notes:
        if not instrument.source(note).has_release:
            raise ValueError(
                f"the signal for {note_name(note)} has no envelope: multiply it by "
                "adsr_envelope(...) so it knows when the key is released, or pass "
                "require_envelope=False to let it fade out over 10 ms instead"
            )


def keyboard(
    function: Callable[[Note], Expr],
    *,
    octave: int = 4,
    gain: float = 0.25,
    require_envelope: bool = True,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    title: str = "daw keyboard",
    silent: bool = False,
    size: tuple[int, int] = WINDOW_SIZE,
) -> None:
    """Open a playable keyboard for a function from a `Note` to a signal.

    The home row plays the white keys and the row above the black keys, `z`
    and `x` move down and up an octave, and the on-screen piano can be clicked.
    Every note is compiled before the window opens, which takes a second or a few.
    A filtered signal cannot be played live and raises a `ValueError`, and so
    does one with no envelope, unless `require_envelope` is false (it then
    fades out over 10 ms when the key is released). The envelope is what
    `adsr_envelope` puts in the signal: it is how a note learns the key came up.
    `gain` scales the sum of the notes (see `Mixer`). With `silent` no sound is
    made and the picture runs on the clock. Returns when the window is closed."""
    instrument = Instrument(function, sample_rate)
    instrument.precompile(NOTES)
    if require_envelope:
        _check_envelope(instrument)
    mixer = Mixer(gain)
    ring = RingBuffer(RING_SAMPLES)
    window = _Keyboard(
        instrument, mixer, ring, octave=octave, title=title, drive_clock=silent
    )
    output = None if silent else LiveOutput(mixer, sample_rate, tap=ring)
    if output is not None:
        output.start()
    try:
        _run(window, size)
    finally:
        if output is not None:
            output.stop()


def keyboard_snapshot(
    function: Callable[[Note], Expr],
    held: tuple[Note, ...],
    output_file: str | os.PathLike[str],
    *,
    octave: int = 4,
    seconds: float = 3.0,
    gain: float = 0.25,
    require_envelope: bool = True,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    title: str = "daw keyboard",
    size: tuple[int, int] = WINDOW_SIZE,
) -> None:
    """Save a picture of the keyboard after `held` notes have been sounding
    for `seconds`. Nothing is opened or played, so it needs no display (the
    docs use it for their pictures). Like `keyboard` it refuses a signal with
    no envelope unless `require_envelope` is false."""
    instrument = Instrument(function, sample_rate)
    instrument.precompile(held)
    if require_envelope:
        _check_envelope(instrument, held)
    mixer = Mixer(gain)
    ring = RingBuffer(RING_SAMPLES)
    window = _Keyboard(instrument, mixer, ring, octave=octave, title=title)
    for note in held:
        semitone = NOTES.index(note) - NOTES.index(Note[f"C{octave}"])
        window.press(note, semitone)
    chunk = 512
    for _ in range(round(seconds * sample_rate) // chunk):
        ring.write(mixer.render(chunk))
        window.advance(0)
    surface = pygame.Surface(size)
    window.draw(surface)
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(path))
