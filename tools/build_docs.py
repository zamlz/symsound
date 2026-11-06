#!/usr/bin/env python3
"""Run the code blocks in docs/ and keep their output and graphs up to date

The documentation pages are markdown processed by `md` (github.com/zamlz/mdutils):

  * a python block followed by `<!-- md-code: id="..."; bin="python3" -->` is
    executed and what it prints is written back under it,
  * the blocks also draw the graphs in docs/img/,
  * `<!-- md-toc: -->` markers get a table of contents.

Usage:
  python tools/build_docs.py           rewrite docs/ in place
  python tools/build_docs.py --check   change nothing; fail if docs/ is stale

--check is what `nix flake check` runs. It rebuilds everything in a scratch
copy and compares text and images with the committed files. A block that raises
is also a failure, because `md` records a traceback as the block's output and
exits successfully.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAILURE_MARKERS = ("Traceback (most recent call last)", "Error: Code execution")


def run_md(command, text, cwd):
    result = subprocess.run(
        ["md", command],
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


def build(docs):
    """Regenerate every page in `docs`. Returns the problems found."""
    problems = []
    for page in sorted(docs.glob("*.md")):
        text = page.read_text()
        built = run_md("toc", run_md("code", text, docs), docs)
        for marker in FAILURE_MARKERS:
            if marker in built:
                problems.append(
                    f"{page.name}: a code block failed\n    {excerpt(built, marker)}"
                )
        page.write_text(built)
    return problems


def referenced_images(docs):
    names = set()
    for page in docs.glob("*.md"):
        for line in page.read_text().splitlines():
            start = line.find("](img/")
            while start != -1:
                end = line.index(")", start)
                names.add(line[start + 2 : end])
                start = line.find("](img/", end)
    return names


def differences(committed, rebuilt):
    problems = []
    for page in sorted(committed.glob("*.md")):
        if page.read_text() != (rebuilt / page.name).read_text():
            problems.append(f"{page.name}: text is out of date")
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
