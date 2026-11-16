# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""Guards that keep the test suite honest as the project grows

These tests do not check behavior. They fail when a change adds code without
the tests that guidelines.md requires, so rigor does not depend on anyone
remembering it.
"""

import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import symsound

TESTS = Path(__file__).parent
ROOT = TESTS.parent
SOURCES = {
    p: p.read_text() for p in TESTS.glob("test_*.py") if p.name != Path(__file__).name
}


def public_api():
    """(module, name) for every public function and class the package defines"""
    found = []
    for info in pkgutil.walk_packages(symsound.__path__, prefix="symsound."):
        module = importlib.import_module(info.name)
        for name, obj in vars(module).items():
            if name.startswith("_"):
                continue
            defined_here = getattr(obj, "__module__", None) == module.__name__
            if defined_here and (inspect.isfunction(obj) or inspect.isclass(obj)):
                found.append((module.__name__, name))
    return found


def test_the_public_api_is_discovered():
    names = {name for _, name in public_api()}
    assert {"sine", "adsr_envelope", "compile", "BPM", "low_pass_filter"} <= names


def test_every_public_function_and_class_is_exercised_by_a_test():
    untested = [
        f"{module}.{name}"
        for module, name in public_api()
        if not any(re.search(rf"\b{name}\b", text) for text in SOURCES.values())
    ]
    assert not untested, (
        f"no test references {untested}. New features need tests that follow "
        "guidelines.md"
    )


def test_every_example_script_is_covered_by_the_example_tests():
    examples = {p.stem for p in (TESTS.parent / "examples").glob("*.py")}
    text = (TESTS / "test_examples.py").read_text()
    missing = [name for name in examples if f'"{name}"' not in text]
    assert not missing, f"examples without tests: {missing}"


def test_known_bugs_are_recorded_as_strict_xfails():
    # A non-strict xfail would silently keep passing after the bug is fixed
    offenders = []
    for path, text in SOURCES.items():
        for match in re.finditer(r"xfail\((.*?)\)\s*\n", text, flags=re.S):
            if "strict=True" not in match.group(1):
                offenders.append(path.name)
    assert not offenders, f"xfail without strict=True in {offenders}"


def test_tests_are_not_silently_skipped():
    offenders = [
        path.name
        for path, text in SOURCES.items()
        if re.search(r"pytest\.mark\.skip\b|@unittest\.skip", text)
    ]
    assert not offenders, f"unconditional skips in {offenders}"


def test_golden_files_exist_for_every_audible_example():
    expected = {
        "c_major_shimmer",
        "echo_melody",
        "echo_pulse",
        "house",
        "house_with_low_pass",
        "keyboard_synth",
        "ode_to_joy",
        "weighted_blend_test",
    }
    present = {p.stem for p in (TESTS / "golden").glob("*.npy")}
    assert expected <= present


LICENSE_LINE = "# SPDX-License-Identifier: GPL-3.0-or-later"
COPYRIGHT_START = "# Copyright (C) "


def source_files():
    """Every source file of the project: the Python in the four code folders,
    and the flake"""
    for folder in ("src", "tests", "examples", "tools"):
        yield from sorted((ROOT / folder).rglob("*.py"))
    yield ROOT / "flake.nix"


def test_every_source_file_is_found():
    names = {p.relative_to(ROOT).as_posix() for p in source_files()}
    assert {
        "src/symsound/__init__.py",
        "src/symsound/waves/utils.py",
        "tests/test_policy.py",
        "examples/house.py",
        "tools/build_docs.py",
        "flake.nix",
    } <= names


def test_every_source_file_starts_with_the_license_header():
    # An optional shebang, then the SPDX line and the copyright line, so a
    # single file copied out of the project still says how it is licensed
    missing = []
    for path in source_files():
        lines = path.read_text().splitlines()
        if lines and lines[0].startswith("#!"):
            lines = lines[1:]
        header = lines[:2]
        if (
            len(header) < 2
            or header[0] != LICENSE_LINE
            or not header[1].startswith(COPYRIGHT_START)
        ):
            missing.append(path.relative_to(ROOT).as_posix())
    assert not missing, (
        f"missing the license header (SPDX line, then a copyright line): {missing}"
    )


def test_the_license_header_matches_the_license_declared_for_the_package():
    declared = re.search(
        r'^license = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M
    )
    assert declared, "pyproject.toml declares no license"
    assert LICENSE_LINE == f"# SPDX-License-Identifier: {declared.group(1)}"
