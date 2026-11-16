"""The equation at the top of README.md is the dry beat of examples/house.py

The LaTeX is written by hand, so this test turns that very text back into a
SymPy expression and compares it with the signal the example builds.
"""

import re
import runpy
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from symsound.waves.time import TIME
from symsound.waves.utils import WHITE_NOISE

ROOT = Path(__file__).parent.parent
SPELLINGS = [
    (r"\left", ""),
    (r"\right", ""),
    (r"\bigl", ""),
    (r"\bigr", ""),
    (r"\Bigl", ""),
    (r"\Bigr", ""),
    (r"\!", ""),
    (r"\,", ""),
    (r"\mathrm{sgn}\sin", "sgnsin"),
    (r"\lfloor", "floor("),
    (r"\rfloor", ")"),
]


def readme_equation():
    text = (ROOT / "README.md").read_text()
    match = re.search(r"\$\$\n(.*?)\n\$\$", text, re.S)
    assert match, "README.md has no $$ equation"
    return match.group(1)


def latex_to_sympy(tex):
    """Handles the small subset of LaTeX the README equation uses"""
    body = tex.split("d(t) ={}&", 1)[1].replace(r"\end{aligned}", "")
    body = body.replace("\\\\", " ").replace("&", " ")
    for old, new in SPELLINGS:
        body = body.replace(old, new)
    for _ in range(5):
        body = re.sub(r"\\tfrac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", body)
    body = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", body)
    for name in ("sin", "cos", "pi"):
        body = body.replace("\\" + name, name)
    body = body.replace("n(t)", "noise")
    assert "\\" not in body, f"unhandled LaTeX left over: {body}"
    t, noise = sp.symbols("t noise")
    sgnsin = sp.Function("sgnsin")
    local = {
        "t": t,
        "noise": noise,
        "floor": sp.floor,
        "sin": sp.sin,
        "cos": sp.cos,
        "pi": sp.pi,
        "sgnsin": sgnsin,
    }
    expr = parse_expr(
        " ".join(body.split()),
        local_dict=local,
        transformations=standard_transformations
        + (implicit_multiplication_application,),
    )
    expr = expr.replace(sgnsin, lambda x: sp.sign(sp.sin(x)))
    return sp.lambdify((t, noise), expr, "numpy")


@pytest.fixture(scope="module")
def dry():
    signal = runpy.run_path(str(ROOT / "examples" / "house.py"))["w"]
    return sp.lambdify((TIME, WHITE_NOISE), signal, "numpy")


def test_the_readme_equation_matches_the_example_signal(dry):
    equation = latex_to_sympy(readme_equation())
    rng = np.random.default_rng(3)
    times = rng.uniform(0, 30, 20000)
    noise = rng.uniform(-1, 1, 20000)
    np.testing.assert_allclose(equation(times, noise), dry(times, noise), atol=1e-9)


def test_the_check_notices_a_wrong_coefficient(dry):
    wrong = readme_equation().replace("0.25", "0.3")
    equation = latex_to_sympy(wrong)
    rng = np.random.default_rng(3)
    times = rng.uniform(0, 30, 20000)
    noise = rng.uniform(-1, 1, 20000)
    assert np.abs(equation(times, noise) - dry(times, noise)).max() > 0.01


def test_the_equation_avoids_macros_github_rejects():
    assert r"\operatorname" not in readme_equation()
