"""Effects: functions from signals to signals, all plain expressions of `TIME`.

* `timing`: `delay`, a signal later (echo and reverb are sums of delays; see
  docs/effects.md).
* `spatial`: `pan`, a mono signal placed between the ears (see docs/stereo.md).

The functions are imported here, so `from daw.effects import delay, pan` is all
anyone needs.
"""

from daw.effects.spatial import pan
from daw.effects.timing import delay

__all__ = ["delay", "pan"]
