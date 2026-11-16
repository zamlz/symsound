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
  python tools/build_docs.py --restore-images LOG
                                       rewrite the images that a failed --check
                                       printed (see below); LOG may be `-`

--check is what `nix flake check` runs. It rebuilds everything in a scratch
copy and compares text and images with the committed files. A block that raises
is also a failure, because `md` records a traceback as the block's output and
exits successfully.

Images are compared byte for byte, and the committed ones are the ones the CI
runner draws: the same graph drawn on another machine (a Mac, say) lands a pixel
or two away, so it would not match. When --check finds images that differ, it also
prints each rebuilt image as base64 between markers, so the runner's version can
be recreated here and committed:

  gh run view RUN_ID --log-failed | python tools/build_docs.py --restore-images -
"""

import argparse
import base64
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
        elif old.read_bytes() != new.read_bytes():
            problems.append(f"{name}: the image is out of date")
    present = {f"img/{p.name}" for p in (committed / "img").glob("*.png")}
    for name in sorted(present - expected):
        problems.append(f"{name}: no page uses this image")
    return problems


def stale_images(committed, rebuilt):
    """The images, as paths inside `docs` (img/name.png), that exist in both
    trees and are not the same file"""
    return [
        name
        for name in sorted(referenced_images(rebuilt))
        if (committed / name).exists()
        and (rebuilt / name).exists()
        and (committed / name).read_bytes() != (rebuilt / name).read_bytes()
    ]


def encoded_images(rebuilt, names, width=76):
    """The images `names` of the tree `rebuilt` as lines of text: each is base64
    between a begin and an end marker, so that a log can carry it"""
    lines = []
    for name in names:
        data = (rebuilt / name).read_bytes()
        text = base64.b64encode(data).decode()
        lines.append(f"=== begin base64 docs/{name} ({len(data)} bytes) ===")
        lines.extend(text[i : i + width] for i in range(0, len(text), width))
        lines.append(f"=== end docs/{name} ===")
    return lines


BEGIN_IMAGE = re.compile(r"=== begin base64 docs/(\S+) \((\d+) bytes\) ===")
END_IMAGE = re.compile(r"=== end docs/(\S+) ===")


def restore_images(text, docs):
    """Write the images that `encoded_images` printed, found in `text` (a CI log
    is fine: each line may start with a timestamp), into `docs`. Returns the
    names written."""
    written = []
    name = size = None
    chunks = []
    for line in text.splitlines():
        line = line.strip()
        begin = BEGIN_IMAGE.search(line)
        if begin:
            name, size, chunks = begin.group(1), int(begin.group(2)), []
        elif name is None:
            continue
        elif END_IMAGE.search(line):
            data = base64.b64decode("".join(chunks))
            if len(data) != size:
                raise SystemExit(f"{name}: expected {size} bytes, got {len(data)}")
            (docs / name).write_bytes(data)
            written.append(name)
            name = None
        elif line:
            chunks.append(line.split()[-1])
    return written


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="verify, change nothing")
    parser.add_argument(
        "--restore-images",
        metavar="LOG",
        help="write the images that a failed --check printed (a file, or - for stdin)",
    )
    args = parser.parse_args(argv)
    if args.restore_images:
        if args.restore_images == "-":
            log = sys.stdin.read()
        else:
            log = Path(args.restore_images).read_text()
        written = restore_images(log, ROOT / "docs")
        for name in written:
            print(f"restored docs/{name}")
        if not written:
            print("no images found in the log", file=sys.stderr)
        return 0 if written else 1
    if shutil.which("md") is None:
        raise SystemExit("md not found: enter the dev shell (nix develop)")

    shown = []
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
            stale = stale_images(ROOT / "docs", copy / "docs")
            shown = encoded_images(copy / "docs", stale)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(
            "run `nix develop -c python tools/build_docs.py` and commit the result",
            file=sys.stderr,
        )
    if args.check and shown:
        print(
            "\nThe rebuilt images that differ from the committed ones, as base64. To "
            "recreate them here:\n  gh run view RUN_ID --log-failed | "
            "python tools/build_docs.py --restore-images -"
        )
        print("\n".join(shown))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
