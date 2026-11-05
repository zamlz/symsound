
import sympy as sp

# Global shared unified time symbol
TIME = sp.symbols('time', real=True, positive=True)


# Beats per Minute
class BPM:
    def __init__(self, bpm: int | float) -> None:
        self._bpm = float(bpm)
        self._bps = self._bpm / 60  # beats per second

    def time(self, beats: int | float) -> float:
        """Return the duration in seconds for float amount of $beats"""
        return beats / self._bps

    def frequency(self, beats: int | float) -> float:
        """Return the frequency of a signal that has a time period of float
        number of $beats"""
        return 1 / self.time(beats)

    def beat(self, time: int | float) -> float:
        """Return the beat number at the given time (provided in seconds)"""
        return self._bps * time
