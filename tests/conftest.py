# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
import os
import sys
import types

import matplotlib
import numpy as np
import pytest
from hypothesis import HealthCheck, settings

matplotlib.use("Agg")

# The visualizer is tested without a screen or a sound card
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Deterministic and un-deadlined for CI, randomized locally to find new bugs
settings.register_profile(
    "default",
    deadline=None,
    max_examples=60,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("ci", settings.get_profile("default"), derandomize=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


class FakeOutputStream:
    """Stands in for sounddevice.OutputStream: tests call the callback by hand"""

    def __init__(self, **options):
        self.options = options
        self.callback = options["callback"]
        self.events: list[str] = []

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")

    def close(self):
        self.events.append("close")


class FakeSoundDevice(types.ModuleType):
    """Stands in for sounddevice so playback can be asserted without audio"""

    def __init__(self) -> None:
        super().__init__("sounddevice")
        self.plays: list[tuple[np.ndarray, int | None]] = []
        self.waits = 0
        self.interrupt = False
        self.streams: list[FakeOutputStream] = []
        self.OutputStream = self._make_stream

    def _make_stream(self, **options):
        stream = FakeOutputStream(**options)
        self.streams.append(stream)
        return stream

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
