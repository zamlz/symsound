# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""The mutation tool's bookkeeping: which tests go with which source file, and
which mutants a nightly run picks

(That the mutants are killed is what running the tool itself shows.)
"""

import datetime
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "mutation_check", ROOT / "tools" / "mutation_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["mutation_check"] = module
    spec.loader.exec_module(module)
    return module


tool = load_tool()


def test_every_mutated_file_has_tests_to_try_first():
    mutated = {relative for _, relative, _, _ in tool.MUTANTS}
    assert mutated <= set(tool.FAST_TESTS), mutated - set(tool.FAST_TESTS)


def test_the_mapped_files_exist():
    for source, tests in tool.FAST_TESTS.items():
        assert (ROOT / source).exists(), source
        for test in tests:
            assert (ROOT / test).exists(), test


def test_every_mutant_text_is_still_in_the_source():
    for label, relative, old, _ in tool.MUTANTS:
        assert old in (ROOT / relative).read_text(), f"stale mutant: {label}"


def test_mutant_labels_are_unique():
    labels = [label for label, *_ in tool.MUTANTS]
    assert len(labels) == len(set(labels))


class TestRotation:
    ITEMS = list(range(50))

    def pick(self, day, count=8):
        return tool.rotation(self.ITEMS, count, datetime.date.fromordinal(day))

    def test_picks_the_requested_number(self):
        assert len(self.pick(739000)) == 8

    def test_asking_for_more_than_exist_runs_them_all(self):
        assert tool.rotation(self.ITEMS, 80) == self.ITEMS

    def test_consecutive_days_pick_different_mutants(self):
        assert set(self.pick(739000)) != set(self.pick(739001))

    def test_a_week_of_nights_covers_every_mutant_in_a_few_rounds(self):
        # 50 mutants at 8 a night: every one comes round within 7 nights
        seen = set()
        for day in range(739000, 739007):
            seen |= set(self.pick(day))
        assert seen == set(self.ITEMS)

    def test_the_same_day_always_picks_the_same_mutants(self):
        assert self.pick(739000) == self.pick(739000)

    @pytest.mark.parametrize("count", [1, 5, 8, 13])
    def test_picks_are_distinct(self, count):
        picked = self.pick(739123, count)
        assert len(set(picked)) == len(picked)


def test_the_full_suite_fallback_skips_the_test_that_fails_for_every_mutant():
    # Regression: that test failing made every mutant look "killed" when the
    # fast tests had missed it
    assert tool.STALE_TEXT_TEST.endswith(
        "::test_every_mutant_text_is_still_in_the_source"
    )
    assert tool.STALE_TEXT_TEST.split("::")[0] == "tests/test_mutation_tool.py"
    assert (
        "test_every_mutant_text_is_still_in_the_source"
        in (ROOT / "tests" / "test_mutation_tool.py").read_text()
    )
