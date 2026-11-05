import pytest

from daw.constants import Note


def test_note_is_a_float():
    assert isinstance(Note.A4, float)
    assert Note.A4 == 440.0


def test_octave_doubles_frequency():
    assert pytest.approx(2 * Note.A4) == Note.A5
    assert pytest.approx(2 * Note.C4, abs=0.01) == Note.C5


def test_notes_are_equal_tempered():
    for note in Note:
        index = list(Note).index(note)
        expected = 27.5 * 2 ** (index / 12)
        assert note.value == pytest.approx(expected, rel=1e-3), note.name
