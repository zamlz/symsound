import re

import pytest

from daw.constants import DEFAULT_SAMPLE_RATE, FloatEnum, Note

SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NAME_PATTERN = re.compile(r"^([A-G])(S?)(\d)$")


def semitones_above_a0(name):
    """Equal-temperament semitone index of a note name, with A0 = 0"""
    letter, sharp, octave = NAME_PATTERN.match(name).groups()
    index_from_c0 = SEMITONES[letter] + (1 if sharp else 0) + 12 * int(octave)
    return index_from_c0 - SEMITONES["A"]


def test_default_sample_rate_is_cd_quality():
    assert DEFAULT_SAMPLE_RATE == 44100


def test_note_is_a_float_enum():
    assert issubclass(Note, FloatEnum)
    assert issubclass(Note, float)


def test_covers_the_88_keys_of_a_piano():
    assert len(Note) == 88
    assert list(Note)[0] is Note.A0
    assert list(Note)[-1] is Note.C8


def test_concert_pitch_reference():
    assert Note.A4 == 440.0


@pytest.mark.parametrize("note", list(Note), ids=lambda n: n.name)
def test_name_is_well_formed(note):
    assert NAME_PATTERN.match(note.name), note.name
    # There is no E# / B# in the table
    assert note.name[0:2] not in {"ES", "BS"}


@pytest.mark.parametrize("note", list(Note), ids=lambda n: n.name)
def test_frequency_matches_equal_temperament(note):
    expected = 440.0 * 2 ** ((semitones_above_a0(note.name) - 48) / 12)
    # Values are rounded to 2 decimals
    assert note.value == pytest.approx(expected, abs=0.006), note.name


def test_table_is_in_ascending_semitone_order():
    names = [n.name for n in Note]
    assert [semitones_above_a0(n) for n in names] == list(range(88))


def test_frequencies_strictly_increase():
    values = [n.value for n in Note]
    assert values == sorted(set(values))


def test_adjacent_ratio_is_the_twelfth_root_of_two():
    values = [n.value for n in Note]
    for low, high in zip(values, values[1:], strict=False):
        assert high / low == pytest.approx(2 ** (1 / 12), rel=2e-3)


OCTAVE_PAIRS = [
    (note.name, f"{note.name[:-1]}{int(note.name[-1]) + 1}")
    for note in Note
    if f"{note.name[:-1]}{int(note.name[-1]) + 1}" in Note.__members__
]


def test_octave_pairs_cover_every_note_with_a_partner_above():
    # Index i has a partner when i + 12 <= 87
    assert len(OCTAVE_PAIRS) == 76


@pytest.mark.parametrize(("low", "high"), OCTAVE_PAIRS, ids=lambda x: str(x))
def test_octave_doubles_frequency(low, high):
    assert Note[high].value == pytest.approx(2 * Note[low].value, rel=2e-3)


def test_names_are_unique_and_lookup_by_value_roundtrips():
    for note in Note:
        assert Note(note.value) is note
        assert Note[note.name] is note
