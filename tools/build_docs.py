#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""Run the code blocks in docs/ and keep their output and graphs up to date

The documentation pages are markdown processed by `md` (github.com/zamlz/mdutils):

  * a python block followed by `<!-- md-code: id="..."; bin="python3" -->` is
    executed and what it prints is written back under it,
  * the blocks also draw the graphs in docs/img/,
  * `<!-- md-toc: -->` markers get a table of contents.

The README at the top of the repository goes through the same two steps. Its
blocks run from the repository root, so it names its graphs `docs/img/...`.

Usage:
  python tools/build_docs.py           rewrite docs/ in place
  python tools/build_docs.py --check   change nothing; fail if docs/ is stale

--check is what `nix flake check` runs. It rebuilds everything in a scratch
copy and compares text and images with the committed files. A block that raises
is also a failure, because `md` records a traceback as the block's output and
exits successfully.

Images are compared by what they look like, not byte for byte. The same graph
drawn on two machines differs by a shade in a few edge pixels (the arithmetic
rounds differently), so identical files are not to be expected. See
`image_difference`.
"""

import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.signal import convolve2d

ROOT = Path(__file__).resolve().parent.parent
FAILURE_MARKERS = ("Traceback (most recent call last)", "Error: Code execution")


def md_environment():
    """`md` runs each block with the first `python3` on PATH. On a machine with
    another Python installed (a CI runner) that is not the one that has the
    project's dependencies, so put this interpreter's directory first."""
    here = str(Path(sys.executable).parent)
    return dict(os.environ, PATH=here + os.pathsep + os.environ.get("PATH", ""))


def run_md(command, text, cwd):
    result = subprocess.run(
        ["md", command],
        env=md_environment(),
        input=text,
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"md {command} failed:\n{result.stderr}")
    return result.stdout


def excerpt(text, marker, lines=14):
    """The marker line and what follows it, so a failure explains itself"""
    after = text[text.index(marker) :].splitlines()[:lines]
    return "\n    ".join(after)


def readme_of(docs):
    """The README beside a docs folder, or None if there is not one"""
    readme = docs.parent / "README.md"
    return readme if readme.exists() else None


def pages_of(docs):
    """Every page that is built, with the folder its blocks run in: the pages
    of `docs`, and the README one level up"""
    pages = [(page, docs) for page in sorted(docs.glob("*.md"))]
    readme = readme_of(docs)
    if readme is not None:
        pages.append((readme, readme.parent))
    return pages


def build_page(page, folder):
    """Regenerate one page in place. Returns the problems found in it."""
    built = run_md("toc", run_md("code", page.read_text(), folder), folder)
    problems = [
        f"{page.name}: a code block failed\n    {excerpt(built, marker)}"
        for marker in FAILURE_MARKERS
        if marker in built
    ]
    page.write_text(built)
    return problems


def build(docs, workers=None):
    """Regenerate every page in `docs`, and the README, `workers` at a time
    (default: one per CPU). Pages share nothing but the folder, so they can run
    side by side; the problems come back in page order."""
    pages = pages_of(docs)
    with ThreadPoolExecutor(max_workers=workers or os.cpu_count() or 1) as pool:
        found = list(pool.map(lambda item: build_page(*item), pages))
    return [problem for page_problems in found for problem in page_problems]


def referenced_images(docs):
    """The images the pages use, as paths inside `docs` (img/name.png). The
    README, which sits above `docs`, writes them as docs/img/name.png."""
    names = set()
    for page, _ in pages_of(docs):
        prefix = "](img/" if page.parent == docs else "](docs/img/"
        for line in page.read_text().splitlines():
            start = line.find(prefix)
            while start != -1:
                end = line.index(")", start)
                names.add("img/" + line[start + len(prefix) : end])
                start = line.find(prefix, end)
    return names


# How different two renderings of the same graph may be. Measured on every image
# that differs between a Mac and a Linux machine: no pixel was off by more than 2
# (of 255), and the sum of the differences in any 5x5 window was at most 10. A real
# change, such as moved text, a different colour or a missing label, scores
# many times more. The limits sit between the two.
MAX_PIXEL_DIFFERENCE = 3
MAX_CLUSTER = 25
CLUSTER_KERNEL = np.ones((5, 5))


def image_difference(old, new):
    """Why two PNG images (as bytes) are not the same picture, or None if they are.

    Identical files are the same. Otherwise they must be the same size, no pixel
    may differ by more than MAX_PIXEL_DIFFERENCE in any colour channel, and the
    differences must not cluster: the kernel of ones in CLUSTER_KERNEL is passed
    over the difference image, which adds up the differences in every 5x5 window,
    and no window may add up to more than MAX_CLUSTER. A few stray pixels that
    each differ by one are rounding. Many that differ close together are a
    change, and the window adds them up.

    Bytes that are not an image are compared as bytes."""
    if old == new:
        return None
    try:
        before = np.asarray(Image.open(io.BytesIO(old)).convert("RGBA"), dtype=int)
        after = np.asarray(Image.open(io.BytesIO(new)).convert("RGBA"), dtype=int)
    except (UnidentifiedImageError, OSError):
        return "the files differ"
    if before.shape != after.shape:
        old_size = f"{before.shape[1]}x{before.shape[0]}"
        new_size = f"{after.shape[1]}x{after.shape[0]}"
        return f"the size changed from {old_size} to {new_size}"
    difference = np.abs(before - after).max(axis=2).astype(float)
    largest = int(difference.max())
    if largest > MAX_PIXEL_DIFFERENCE:
        return f"a pixel differs by {largest} (at most {MAX_PIXEL_DIFFERENCE} allowed)"
    cluster = float(convolve2d(difference, CLUSTER_KERNEL, mode="same").max())
    if cluster > MAX_CLUSTER:
        return (
            f"the differences cluster: {cluster:.0f} in a 5x5 window "
            f"(at most {MAX_CLUSTER} allowed)"
        )
    return None


def differences(committed, rebuilt):
    problems = []
    for page in sorted(committed.glob("*.md")):
        if page.read_text() != (rebuilt / page.name).read_text():
            problems.append(f"{page.name}: text is out of date")
    old_readme, new_readme = readme_of(committed), readme_of(rebuilt)
    if (
        old_readme is not None
        and new_readme is not None
        and old_readme.read_text() != new_readme.read_text()
    ):
        problems.append("README.md: text is out of date")
    expected = referenced_images(rebuilt)
    for name in sorted(expected):
        old, new = committed / name, rebuilt / name
        if not new.exists():
            problems.append(f"{name}: referenced but no block draws it")
        elif not old.exists():
            problems.append(f"{name}: not committed")
        else:
            reason = image_difference(old.read_bytes(), new.read_bytes())
            if reason is not None:
                problems.append(f"{name}: the image is out of date ({reason})")
    present = {f"img/{p.name}" for p in (committed / "img").glob("*.png")}
    for name in sorted(present - expected):
        problems.append(f"{name}: no page uses this image")
    return problems


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="verify, change nothing")
    args = parser.parse_args(argv)
    if shutil.which("md") is None:
        raise SystemExit("md not found: enter the dev shell (nix develop)")

    if not args.check:
        problems = build(ROOT / "docs")
        referenced = referenced_images(ROOT / "docs")
        for stale in (ROOT / "docs" / "img").glob("*.png"):
            if f"img/{stale.name}" not in referenced:
                stale.unlink()
                print(f"removed unused {stale.relative_to(ROOT)}")
    else:
        with tempfile.TemporaryDirectory() as scratch:
            copy = Path(scratch)
            shutil.copytree(ROOT / "docs", copy / "docs")
            shutil.copytree(ROOT / "examples", copy / "examples")
            shutil.copy2(ROOT / "README.md", copy / "README.md")
            problems = build(copy / "docs")
            problems += differences(ROOT / "docs", copy / "docs")
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(
            "run `nix develop -c python tools/build_docs.py` and commit the result",
            file=sys.stderr,
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
