# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""The documentation in docs/ is tested like code

docs/ pages are processed by `md` (github.com/zamlz/mdutils): python blocks
followed by an `md-code` comment are executed, their output is written back
under them, and they draw the graphs in docs/img/. `python tools/build_docs.py`
does that for every page, and `--check` fails when anything is stale.

These tests check the structure of the pages (cheap, no md needed) and, when
md is installed, run the real thing. The flake runs the md part as its own
check, `docs`, because the pytest check has no md.
"""

import ast
import builtins
import importlib.util
import io
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

from test_policy import public_api

ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"
PAGES = sorted(DOCS.glob("*.md"))
WALKTHROUGH = DOCS / "house-with-low-pass-walkthrough.md"
KICK = (ROOT / "examples" / "house_with_low_pass.py").read_text()

FENCED_PYTHON = re.compile(
    r"^```python([^\n]*)\n(.*?)^```\n(<!-- md-code: ([^>]*) -->)?", re.S | re.M
)
DIRECTIVE_FIELD = re.compile(r'(\w+)=(?:"([^"]*)"|(\d+))')

# Implementation details that users never call
INTERNAL = {"FloatEnum"}


def python_blocks(page):
    """(info string, code, directive fields or None) of every python block"""
    found = []
    for info, code, _, directive in FENCED_PYTHON.findall(page.read_text()):
        fields = None
        if directive:
            fields = {k: a or b for k, a, b in DIRECTIVE_FIELD.findall(directive)}
        found.append((info.strip(), code, fields))
    return found


def load_build_docs():
    spec = importlib.util.spec_from_file_location(
        "build_docs", ROOT / "tools" / "build_docs.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_docs"] = module
    spec.loader.exec_module(module)
    return module


def test_there_are_docs_to_check():
    names = {p.name for p in PAGES}
    assert {"README.md", "concepts.md", "house-with-low-pass-walkthrough.md"} <= names


class TestCodeBlocks:
    @pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
    def test_every_python_block_is_executed_quoted_or_marked_no_run(self, page):
        for info, code, fields in python_blocks(page):
            if fields is not None or "no-run" in info:
                continue
            assert code.strip() in KICK, (
                f"{page.name}: a python block is neither run by md (add an "
                f"md-code comment), quoted from examples/house.py, nor marked "
                f"`python no-run`:\n{code}"
            )

    def test_every_python_block_in_the_readme_is_executed_or_marked_no_run(self):
        # The README is built with the docs, so a block that is neither run by md
        # nor marked `python no-run` would show code nobody ever checks
        blocks = python_blocks(ROOT / "README.md")
        assert blocks
        for info, code, fields in blocks:
            assert fields is not None or "no-run" in info, (
                f"README.md: a python block is neither run by md nor marked "
                f"`python no-run`:\n{code}"
            )

    @pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
    def test_executed_blocks_have_a_unique_id_and_run_python(self, page):
        ids = []
        for _, code, fields in python_blocks(page):
            if fields is None:
                continue
            assert fields.get("id"), f"{page.name}: md-code needs an id\n{code}"
            assert fields.get("bin") == "python3", f"{page.name}: bin must be python3"
            ids.append(fields["id"])
        assert len(ids) == len(set(ids)), f"{page.name} repeats a block id"

    @pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
    def test_every_executed_block_is_self_contained(self, page):
        # md runs each block in a fresh interpreter, so a block must import or
        # define everything it uses
        for _, code, fields in python_blocks(page):
            if fields is None:
                continue
            bound, loaded = set(dir(builtins)), set()
            for node in ast.walk(ast.parse(code)):
                if isinstance(node, ast.Name):
                    (loaded if isinstance(node.ctx, ast.Load) else bound).add(node.id)
                elif isinstance(node, ast.Import | ast.ImportFrom):
                    bound.update((a.asname or a.name).split(".")[0] for a in node.names)
                elif isinstance(node, ast.arg):
                    bound.add(node.arg)
                elif isinstance(node, ast.FunctionDef | ast.ClassDef):
                    bound.add(node.name)
            assert loaded <= bound, (
                f"{page.name}: uses {sorted(loaded - bound)}\n{code}"
            )

    @pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
    def test_recorded_output_has_no_tracebacks(self, page):
        text = page.read_text()
        assert "Traceback (most recent call last)" not in text
        assert "Error: Code execution" not in text


class TestImages:
    def test_every_referenced_image_exists(self):
        build_docs = load_build_docs()
        names = build_docs.referenced_images(DOCS)
        assert names, "the docs should have graphs"
        missing = [n for n in sorted(names) if not (DOCS / n).exists()]
        assert not missing, f"run tools/build_docs.py: missing {missing}"

    def test_no_image_is_left_unused(self):
        build_docs = load_build_docs()
        used = build_docs.referenced_images(DOCS)
        unused = [
            p.name for p in (DOCS / "img").glob("*.png") if f"img/{p.name}" not in used
        ]
        assert not unused, f"unused images: {unused}"

    def test_images_are_real_pngs_of_a_sensible_size(self):
        import matplotlib.pyplot as plt

        for path in (DOCS / "img").glob("*.png"):
            image = plt.imread(path)
            assert image.shape[1] >= 600 and image.shape[0] >= 250, path.name
            assert image[..., :3].std() > 0.03, f"{path.name} looks blank"

    def test_every_image_has_alt_text(self):
        for page in PAGES:
            for alt in re.findall(r"!\[([^\]]*)\]\(img/", page.read_text()):
                assert alt.strip(), f"{page.name} has an image without alt text"


class TestLinks:
    @pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
    def test_relative_links_point_at_real_pages(self, page):
        for target in re.findall(r"\]\(([^)#:]+\.md)(?:#[^)]*)?\)", page.read_text()):
            assert (page.parent / target).exists(), f"{page.name} links to {target}"

    def test_the_index_lists_every_page(self):
        index = (DOCS / "README.md").read_text()
        others = [p.name for p in PAGES if p.name != "README.md"]
        missing = [name for name in others if name not in index]
        assert not missing, f"docs/README.md does not link {missing}"


class TestCoverage:
    def test_every_public_function_and_class_is_documented(self):
        text = "\n".join(p.read_text() for p in PAGES)
        undocumented = [
            f"{module}.{name}"
            for module, name in public_api()
            if name not in INTERNAL and not re.search(rf"\b{name}\b", text)
        ]
        assert not undocumented, f"add {undocumented} to the pages in docs/"


class TestKickWalkthrough:
    def quoted(self):
        return [
            code for info, code, fields in python_blocks(WALKTHROUGH) if fields is None
        ]

    def test_quotes_the_real_example(self):
        assert self.quoted()
        for code in self.quoted():
            assert code.strip() in KICK, f"not in examples/house.py:\n{code}"

    def test_covers_the_whole_example(self):
        # Every code line of house.py (apart from blank lines, comments and the
        # shebang) must appear in a quoted block, so edits to the example are
        # noticed here and the docs get updated
        quoted = "\n".join(self.quoted())
        lines = [
            line
            for line in KICK.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        missing = [line for line in lines if line not in quoted]
        assert not missing, f"{WALKTHROUGH.name} is missing {missing}"

    def test_the_graph_blocks_load_the_example_without_playing_it(self):
        # run_path without run_name="__main__" must build the signals only
        assert 'if __name__ == "__main__":' in KICK
        graph_blocks = [
            code for _, code, fields in python_blocks(WALKTHROUGH) if fields is not None
        ]
        assert graph_blocks
        assert all("run_path" in code for code in graph_blocks)


class TestBuildDocsTool:
    @pytest.fixture
    def tool(self):
        return load_build_docs()

    def test_md_runs_blocks_with_this_interpreter_not_the_first_python_on_path(
        self, tool, monkeypatch
    ):
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        path = tool.md_environment()["PATH"].split(":")
        assert path[0] == str(Path(sys.executable).parent)
        assert path[1:] == ["/usr/bin", "/bin"]

    def test_a_failure_shows_the_error_not_just_that_there_was_one(self, tool):
        text = "fine\nTraceback (most recent call last)\n  File x\nValueError: boom\n"
        shown = tool.excerpt(text, "Traceback (most recent call last)")
        assert "ValueError: boom" in shown
        assert "fine" not in shown

    def test_finds_every_image_reference_on_a_line(self, tool, tmp_path):
        (tmp_path / "a.md").write_text(
            "![x](img/one.png) text ![y](img/two.png)\n[link](other.md)\n"
        )
        assert tool.referenced_images(tmp_path) == {"img/one.png", "img/two.png"}

    def make(self, root, text, images):
        (root / "img").mkdir(parents=True)
        (root / "p.md").write_text(text)
        for name, data in images.items():
            (root / "img" / name).write_bytes(data)

    def test_identical_trees_have_no_differences(self, tool, tmp_path):
        for side in ("old", "new"):
            self.make(tmp_path / side, "![a](img/a.png)\n", {"a.png": b"1"})
        assert tool.differences(tmp_path / "old", tmp_path / "new") == []

    def test_reports_stale_text_and_images_and_orphans(self, tool, tmp_path):
        self.make(
            tmp_path / "old",
            "old text ![a](img/a.png)\n",
            {"a.png": b"1", "orphan.png": b"2"},
        )
        self.make(tmp_path / "new", "new text ![a](img/a.png)\n", {"a.png": b"9"})
        problems = "\n".join(tool.differences(tmp_path / "old", tmp_path / "new"))
        assert "p.md: text is out of date" in problems
        assert "img/a.png: the image is out of date" in problems
        assert "img/orphan.png: no page uses this image" in problems

    def test_reports_images_nothing_draws_and_images_not_committed(
        self, tool, tmp_path
    ):
        self.make(
            tmp_path / "old", "![a](img/a.png) ![b](img/b.png)\n", {"b.png": b"1"}
        )
        self.make(
            tmp_path / "new", "![a](img/a.png) ![b](img/b.png)\n", {"a.png": b"1"}
        )
        problems = "\n".join(tool.differences(tmp_path / "old", tmp_path / "new"))
        assert "img/a.png: not committed" in problems
        assert "img/b.png: referenced but no block draws it" in problems

    def readme_tree(self, root, readme, page="p.md"):
        (root / "docs" / "img").mkdir(parents=True)
        (root / "docs" / page).write_text("![a](img/a.png)\n")
        (root / "docs" / "img" / "a.png").write_bytes(b"1")
        (root / "README.md").write_text(readme)
        return root / "docs"

    def test_the_readme_is_built_with_the_docs_from_the_repository_root(
        self, tool, tmp_path
    ):
        docs = self.readme_tree(tmp_path, "# symsound\n")
        pages = tool.pages_of(docs)
        assert [(page.name, folder) for page, folder in pages] == [
            ("p.md", docs),
            ("README.md", tmp_path),
        ]

    def test_there_is_no_readme_page_when_there_is_no_readme(self, tool, tmp_path):
        docs = self.readme_tree(tmp_path, "x")
        (tmp_path / "README.md").unlink()
        assert [page.name for page, _ in tool.pages_of(docs)] == ["p.md"]

    def test_images_the_readme_uses_count_as_used(self, tool, tmp_path):
        docs = self.readme_tree(
            tmp_path, "![t](docs/img/readme-tone.png) ![u](docs/img/u.png)\n"
        )
        assert tool.referenced_images(docs) == {
            "img/a.png",
            "img/readme-tone.png",
            "img/u.png",
        }

    def test_a_readme_image_path_is_not_read_as_a_docs_image(self, tool, tmp_path):
        docs = self.readme_tree(tmp_path, "![t](img/not-docs-relative.png)\n")
        assert tool.referenced_images(docs) == {"img/a.png"}

    def test_a_stale_readme_is_reported(self, tool, tmp_path):
        old = self.readme_tree(tmp_path / "old", "old readme\n")
        new = self.readme_tree(tmp_path / "new", "new readme\n")
        problems = "\n".join(tool.differences(old, new))
        assert "README.md: text is out of date" in problems

    def test_an_up_to_date_readme_is_not_reported(self, tool, tmp_path):
        old = self.readme_tree(tmp_path / "old", "same\n")
        new = self.readme_tree(tmp_path / "new", "same\n")
        assert tool.differences(old, new) == []

    def test_a_readme_image_nobody_draws_is_reported(self, tool, tmp_path):
        old = self.readme_tree(tmp_path / "old", "![t](docs/img/t.png)\n")
        (old / "img" / "t.png").write_bytes(b"1")
        new = self.readme_tree(tmp_path / "new", "![t](docs/img/t.png)\n")
        problems = "\n".join(tool.differences(old, new))
        assert "img/t.png: referenced but no block draws it" in problems

    def fake_md(self, tool, monkeypatch, barrier=None):
        """`md` replaced by something that marks what it was asked to do"""

        def run(command, text, cwd):
            if barrier is not None:
                barrier.wait(timeout=10)
            if command == "code":
                return text + (
                    "Traceback (most recent call last)\n" if "bad" in text else ""
                )
            return text + "<toc>\n"

        monkeypatch.setattr(tool, "run_md", run)

    def pages(self, root, names):
        root.mkdir()
        for name in names:
            (root / f"{name}.md").write_text(f"page {name}\n")
        return root

    def test_build_rewrites_every_page_with_the_code_then_the_toc_step(
        self, tool, tmp_path, monkeypatch
    ):
        self.fake_md(tool, monkeypatch)
        docs = self.pages(tmp_path / "docs", ["a", "b", "c"])
        assert tool.build(docs) == []
        for name in "abc":
            assert (docs / f"{name}.md").read_text() == f"page {name}\n<toc>\n"

    def test_build_also_builds_the_readme_from_the_folder_above_docs(
        self, tool, tmp_path, monkeypatch
    ):
        ran_in = []

        def run(command, text, cwd):
            ran_in.append((command, text.strip().split("\n")[0], cwd))
            return text

        monkeypatch.setattr(tool, "run_md", run)
        docs = self.pages(tmp_path / "docs", ["a"])
        (tmp_path / "README.md").write_text("readme\n")
        assert tool.build(docs, workers=1) == []
        assert ("code", "readme", tmp_path) in ran_in
        assert ("toc", "readme", tmp_path) in ran_in
        assert ("code", "page a", docs) in ran_in

    def test_build_reports_a_failed_block_by_page_in_page_order(
        self, tool, tmp_path, monkeypatch
    ):
        self.fake_md(tool, monkeypatch)
        docs = self.pages(tmp_path / "docs", ["a", "bad-one", "c", "bad-two"])
        problems = tool.build(docs)
        assert [p.split(":")[0] for p in problems] == ["bad-one.md", "bad-two.md"]
        assert all("Traceback (most recent call last)" in p for p in problems)

    def test_build_runs_pages_side_by_side(self, tool, tmp_path, monkeypatch):
        # Each fake md call waits for a second one, so this only finishes if two
        # pages are being built at the same moment
        self.fake_md(tool, monkeypatch, barrier=threading.Barrier(2))
        docs = self.pages(tmp_path / "docs", ["a", "b"])
        assert tool.build(docs, workers=2) == []

    def test_build_with_one_worker_gives_the_same_result(
        self, tool, tmp_path, monkeypatch
    ):
        self.fake_md(tool, monkeypatch)
        names = ["a", "bad", "c"]
        many = self.pages(tmp_path / "many", names)
        one = self.pages(tmp_path / "one", names)
        assert tool.build(many, workers=3) == tool.build(one, workers=1)
        for name in names:
            assert (many / f"{name}.md").read_text() == (one / f"{name}.md").read_text()

    def test_a_failure_of_md_itself_stops_the_build(self, tool, tmp_path, monkeypatch):
        def broken(command, text, cwd):
            raise SystemExit("md code failed")

        monkeypatch.setattr(tool, "run_md", broken)
        docs = self.pages(tmp_path / "docs", ["a", "b"])
        with pytest.raises(SystemExit, match="md code failed"):
            tool.build(docs)


def png(pixels, compression=6):
    """The bytes of a PNG of an array of pixels (rows, columns, 3 or 4)"""
    mode = "RGBA" if pixels.shape[2] == 4 else "RGB"
    out = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8), mode).save(
        out, "PNG", compress_level=compression
    )
    return out.getvalue()


def drawing(width=120, height=80):
    """A white picture with a black rectangle, a diagonal line and some 'text'"""
    pixels = np.full((height, width, 3), 255)
    pixels[10:30, 10:50] = 0
    for i in range(min(width, height)):
        pixels[i, i] = 0
    rng = np.random.default_rng(1)
    pixels[50:70, 60:110] = np.where(rng.random((20, 50, 1)) < 0.5, 0, 255)
    return pixels


def nudged(pixels, rows, columns, amount):
    """The picture with a block of it changed by `amount` in every channel (the
    drawings are mostly white, so use a negative amount to see a change)"""
    out = pixels.copy()
    out[rows, columns] = np.clip(out[rows, columns] + amount, 0, 255)
    return out


class TestImageDifference:
    """Two drawings of one graph by different machines differ by rounding in a
    few edge pixels; a real change does not look like that"""

    @pytest.fixture
    def tool(self):
        return load_build_docs()

    def test_the_same_file_is_the_same_picture(self, tool):
        data = png(drawing())
        assert tool.image_difference(data, data) is None

    def test_the_same_pixels_saved_differently_are_the_same_picture(self, tool):
        pixels = drawing()
        fast, small = png(pixels, compression=1), png(pixels, compression=9)
        assert fast != small
        assert tool.image_difference(fast, small) is None

    def test_scattered_single_shades_of_rounding_are_the_same_picture(self, tool):
        pixels = drawing(846, 392)
        rng = np.random.default_rng(0)
        noisy = pixels.copy()
        # As many pixels as the worst measured pair (745), each off by one shade
        rows = rng.integers(0, 392, 745)
        columns = rng.integers(0, 846, 745)
        noisy[rows, columns] = np.clip(noisy[rows, columns] - 1, 0, 255)
        assert tool.image_difference(png(pixels), png(noisy)) is None

    def test_the_limits_are_where_the_measured_rounding_noise_is_below_them(self, tool):
        # Measured over every image that differs between a Mac and Linux: no pixel
        # off by more than 2, and no 5x5 window adding up to more than 10
        assert tool.MAX_PIXEL_DIFFERENCE >= 2
        assert tool.MAX_CLUSTER >= 10
        assert tool.CLUSTER_KERNEL.shape == (5, 5)
        assert (tool.CLUSTER_KERNEL == 1).all()

    def test_a_window_full_of_single_shades_is_at_the_limit_and_allowed(self, tool):
        pixels = drawing()
        block = nudged(pixels, slice(40, 45), slice(20, 25), -1)  # 25 pixels
        assert tool.image_difference(png(pixels), png(block)) is None

    def test_a_window_of_larger_shades_is_a_change(self, tool):
        pixels = drawing()
        block = nudged(pixels, slice(40, 45), slice(20, 25), -2)  # 25 x 2 = 50
        reason = tool.image_difference(png(pixels), png(block))
        assert "cluster" in reason
        assert "50" in reason

    def test_a_small_block_of_bigger_shades_adds_up_to_a_change(self, tool):
        pixels = drawing()
        block = nudged(pixels, slice(40, 43), slice(20, 23), -3)  # 9 x 3 = 27
        assert "cluster" in tool.image_difference(png(pixels), png(block))

    def test_the_same_shades_spread_apart_do_not_add_up(self, tool):
        # 25 pixels of 1 each, but six apart: no window holds more than one
        pixels = drawing(120, 80)
        spread = pixels.copy()
        for row in range(5, 80, 15):
            for column in range(5, 120, 24):
                spread[row, column] = np.clip(spread[row, column] - 1, 0, 255)
        assert tool.image_difference(png(pixels), png(spread)) is None

    def test_a_clustered_difference_is_caught_that_a_biggest_pixel_test_misses(
        self, tool
    ):
        # Every pixel of a 6x6 patch is off by 3: each is within the per-pixel
        # limit, but together they are a visible change
        pixels = drawing()
        block = nudged(pixels, slice(40, 46), slice(20, 26), -3)
        assert tool.image_difference(png(pixels), png(block)) is not None

    @given(
        st.integers(0, 79), st.integers(0, 119), st.integers(-3, 3), st.integers(0, 2)
    )
    def test_one_pixel_off_by_up_to_three_shades_is_allowed(
        self, row, column, amount, channel
    ):
        tool = load_build_docs()
        pixels = np.full((80, 120, 3), 128)
        changed = pixels.copy()
        changed[row, column, channel] = np.clip(
            changed[row, column, channel] + amount, 0, 255
        )
        assert tool.image_difference(png(pixels), png(changed)) is None

    @pytest.mark.parametrize("amount", [4, 10, 255])
    def test_one_pixel_off_by_more_than_three_shades_is_a_change(self, tool, amount):
        pixels = np.zeros((40, 40, 3))
        changed = pixels.copy()
        changed[20, 20, 1] += amount
        reason = tool.image_difference(png(pixels), png(changed))
        assert f"a pixel differs by {amount}" in reason

    def test_text_moved_by_one_pixel_is_a_change(self, tool):
        pixels = drawing()
        moved = pixels.copy()
        moved[50:70, 60:110] = np.roll(pixels[50:70, 60:110], 1, axis=1)
        assert tool.image_difference(png(pixels), png(moved)) is not None

    def test_a_different_size_is_a_change(self, tool):
        reason = tool.image_difference(png(drawing(120, 80)), png(drawing(121, 80)))
        assert "size changed from 120x80 to 121x80" in reason

    def test_the_transparency_counts_too(self, tool):
        pixels = np.dstack([drawing(), np.full((80, 120), 255)])
        changed = pixels.copy()
        changed[30, 30, 3] = 100
        assert "a pixel differs by 155" in tool.image_difference(
            png(pixels), png(changed)
        )

    def test_bytes_that_are_not_images_are_compared_as_bytes(self, tool):
        assert tool.image_difference(b"not an image", b"not an image") is None
        assert tool.image_difference(b"one", b"two") == "the files differ"

    @given(st.integers(0, 2**32 - 1), st.integers(1, 40))
    def test_it_does_not_matter_which_picture_is_which(self, seed, count):
        tool = load_build_docs()
        rng = np.random.default_rng(seed)
        pixels = rng.integers(0, 256, (16, 16, 3))
        changed = pixels.copy()
        rows, columns = rng.integers(0, 16, count), rng.integers(0, 16, count)
        changed[rows, columns] = np.clip(
            changed[rows, columns] + rng.integers(-5, 6, (count, 3)), 0, 255
        )
        forwards = tool.image_difference(png(pixels), png(changed))
        backwards = tool.image_difference(png(changed), png(pixels))
        assert (forwards is None) == (backwards is None)


class TestImagesInTheCheck:
    @pytest.fixture
    def tool(self):
        return load_build_docs()

    def tree(self, root, pixels):
        (root / "img").mkdir(parents=True)
        (root / "p.md").write_text("![a](img/a.png)\n")
        (root / "img" / "a.png").write_bytes(png(pixels))
        return root

    def test_an_image_that_differs_by_rounding_is_up_to_date(self, tool, tmp_path):
        pixels = drawing()
        rounded = nudged(pixels, slice(30, 31), slice(30, 40), -1)
        old = self.tree(tmp_path / "old", pixels)
        new = self.tree(tmp_path / "new", rounded)
        assert tool.differences(old, new) == []

    def test_an_image_that_really_changed_is_reported_with_the_reason(
        self, tool, tmp_path
    ):
        pixels = drawing()
        old = self.tree(tmp_path / "old", pixels)
        new = self.tree(
            tmp_path / "new", nudged(pixels, slice(40, 46), slice(20, 26), -60)
        )
        problems = tool.differences(old, new)
        assert len(problems) == 1
        assert problems[0].startswith("img/a.png: the image is out of date (")
        assert "pixel differs by 60" in problems[0]


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("md") is None, reason="md (mdutils) is not installed")
def test_docs_are_up_to_date_with_what_their_code_produces():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_docs.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
