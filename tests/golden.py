# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""Golden-file support: coarse fingerprints of the signals the examples make

A fingerprint is every PRIME-th sample of the compiled signal. It is small to
store, and still changes whenever the sound changes. Regenerate deliberately
with `UPDATE_GOLDEN=1 pytest tests/test_examples.py` and review the diff of
tests/golden/ like any other change to the sound of the project.
"""

import os
from pathlib import Path

import numpy as np

GOLDEN_DIR = Path(__file__).parent / "golden"
PRIME = 997


def fingerprint(signal):
    return np.asarray(signal, dtype=np.float64)[::PRIME]


def check_golden(name, signal, atol=1e-6):
    path = GOLDEN_DIR / f"{name}.npy"
    current = fingerprint(signal)
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN_DIR.mkdir(exist_ok=True)
        np.save(path, current)
        return
    assert path.exists(), f"missing golden file {path}; run with UPDATE_GOLDEN=1"
    expected = np.load(path)
    assert current.shape == expected.shape, "signal length changed"
    np.testing.assert_allclose(current, expected, atol=atol)
