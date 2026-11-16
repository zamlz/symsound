# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 Amlesh Sivanantham
"""Effects: functions from signals to signals, all plain expressions of `TIME`.

* `timing`: `delay`, a signal later (echo and reverb are sums of delays; see
  docs/effects.md).
* `spatial`: `pan`, a mono signal placed between the ears (see docs/stereo.md).

The functions are imported here, so `from symsound.effects import delay, pan` is all
anyone needs.
"""

from symsound.effects.spatial import pan
from symsound.effects.timing import delay

__all__ = ["delay", "pan"]
