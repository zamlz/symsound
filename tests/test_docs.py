"""The documentation in docs/ is tested like code

Every python block is executed, so examples cannot rot. The blocks of one page
share a namespace and run in order. The kick.py walkthrough must quote the
real example verbatim. Every public name must be documented.
"""

import re
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from test_policy import public_api

ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"
PAGES = sorted(DOCS.glob("*.md"))
BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)

# Implementation details that users never call
INTERNAL = {"FloatEnum"}


def blocks(page):
    return BLOCK.findall(page.read_text())


def test_there_are_docs_to_check():
    names = {p.name for p in PAGES}
    assert {"README.md", "concepts.md", "kick-walkthrough.md"} <= names


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_python_block_runs(page, fake_sounddevice, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)  # no window
    np.random.seed(0)
    namespace = {"__name__": "__docs__"}
    for number, code in enumerate(blocks(page), start=1):
        try:
            exec(compile(code, f"{page.name} block {number}", "exec"), namespace)
        except Exception as error:
            raise AssertionError(
                f"{page.name} python block {number} failed: {error!r}\n{code}"
            ) from error


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_relative_links_point_at_real_pages(page):
    for target in re.findall(r"\]\(([^)#:]+\.md)(?:#[^)]*)?\)", page.read_text()):
        assert (page.parent / target).exists(), f"{page.name} links to {target}"


def test_the_index_lists_every_page():
    index = (DOCS / "README.md").read_text()
    missing = [p.name for p in PAGES if p.name != "README.md" and p.name not in index]
    assert not missing, f"docs/README.md does not link {missing}"


def test_every_public_function_and_class_is_documented():
    text = "\n".join(p.read_text() for p in PAGES)
    undocumented = [
        f"{module}.{name}"
        for module, name in public_api()
        if name not in INTERNAL and not re.search(rf"\b{name}\b", text)
    ]
    assert not undocumented, f"add {undocumented} to the pages in docs/"


def test_the_kick_walkthrough_quotes_the_real_example():
    source = (ROOT / "examples" / "kick.py").read_text()
    quoted = blocks(DOCS / "kick-walkthrough.md")
    assert quoted
    for code in quoted:
        assert code.strip() in source, f"not in examples/kick.py:\n{code}"


def test_the_kick_walkthrough_covers_the_whole_example():
    # Every code line of kick.py (apart from blank lines, comments and the
    # shebang) must appear in a quoted block, so edits to the example are
    # noticed here and the docs get updated
    quoted = "\n".join(blocks(DOCS / "kick-walkthrough.md"))
    lines = [
        line
        for line in (ROOT / "examples" / "kick.py").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    missing = [line for line in lines if line not in quoted]
    assert not missing, f"docs/kick-walkthrough.md is missing {missing}"


@pytest.mark.slow
def test_the_walkthrough_blocks_rebuild_the_example_signal(
    fake_sounddevice, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__docs__"}
    for code in blocks(DOCS / "kick-walkthrough.md"):
        exec(code, namespace)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    real = runpy.run_path(str(ROOT / "examples" / "kick.py"), run_name="__main__")
    assert namespace["w"] == real["w"]
