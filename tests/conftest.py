import os
import sys
import types

import matplotlib
import numpy as np
import pytest
from hypothesis import HealthCheck, settings

matplotlib.use("Agg")

# Deterministic and un-deadlined for CI, randomized locally to find new bugs
settings.register_profile(
    "default",
    deadline=None,
    max_examples=60,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("ci", settings.get_profile("default"), derandomize=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


class FakeSoundDevice(types.ModuleType):
    """Stands in for sounddevice so playback can be asserted without audio"""

    def __init__(self) -> None:
        super().__init__("sounddevice")
        self.plays: list[tuple[np.ndarray, int | None]] = []
        self.waits = 0
        self.interrupt = False

    def play(self, data, samplerate=None):
        if self.interrupt:
            raise KeyboardInterrupt
        self.plays.append((np.asarray(data), samplerate))

    def wait(self):
        self.waits += 1


@pytest.fixture
def fake_sounddevice(monkeypatch):
    fake = FakeSoundDevice()
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return fake


@pytest.fixture(autouse=True)
def close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
