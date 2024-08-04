from __future__ import annotations

import abc
import operator
from typing import Union

import numpy as np
import sympy as sp
from sympy.core.expr import Expr

from daw.constants import DEFAULT_SAMPLE_RATE, Note


TIME = sp.symbols('time')


class AbstractWaveFunction(abc.ABC):
    abc.abstractproperty
    def function(self) -> Expr:
        ...

    def compile(
            self,
            duration: float, 
            sample_rate: int = DEFAULT_SAMPLE_RATE
    ) -> np.ndarray:
        divisions = int(sample_rate * duration)
        linspace = np.linspace(0, duration, divisions, endpoint=False)
        equation = sp.lambdify(TIME, self.function, "numpy")
        return equation(linspace)

    @staticmethod
    def _op_handler(
            left: Union[AbstractWaveFunction, float, int],
            right: Union[AbstractWaveFunction, float, int],
            op_function: Callable
    ) -> WaveFunction:
        x = left.function if isinstance(left, AbstractWaveFunction) else left
        y = right.function if isinstance(right, AbstractWaveFunction) else right
        return WaveFunction(op_function(x, y))

    def __add__(
            self,
            other: Union[AbstractWaveFunction, float, int]
    ) -> WaveFunction:
        return AbstractWaveFunction._op_handler(self, other, operator.add)

    def __radd__(
            self,
            other: Union[AbstractWaveFunction, float, int]
    ) -> WaveFunction:
        return AbstractWaveFunction._op_handler(other, self, operator.add)
    
    def __sub__(
            self,
            other: Union[AbstractWaveFunction, float, int]
    ) -> WaveFunction:
        return AbstractWaveFunction._op_handler(self, other, operator.sub)

    def __rsub__(
            self,
            other: Union[AbstractWaveFunction, float, int]
    ) -> WaveFunction:
        return AbstractWaveFunction._op_handler(other, self, operator.sub)

    def __mul__(
            self,
            other: Union[AbstractWaveFunction, float, int]
    ) -> WaveFunction:
        return AbstractWaveFunction._op_handler(self, other, operator.mul)

    def __rmul__(
            self,
            other: Union[AbstractWaveFunction, float, int]
    ) -> WaveFunction:
        return AbstractWaveFunction._op_handler(other, self, operator.mul)

    def __truediv__(
            self,
            other: Union[AbstractWaveFunction, float, int]
    ) -> WaveFunction:
        return AbstractWaveFunction._op_handler(self, other, operator.truediv)

    def __rtruediv__(
            self,
            other: Union[AbstractWaveFunction, float, int]
    ) -> WaveFunction:
        return AbstractWaveFunction._op_handler(other, self, operator.truediv)

    def __floordiv__(
            self,
            other: Union[AbstractWaveFunction, float, int]
    ) -> WaveFunction:
        return AbstractWaveFunction._op_handler(self, other, operator.floordiv)

    def __rfloordiv__(
            self,
            other: Union[AbstractWaveFunction, float, int]
    ) -> WaveFunction:
        return AbstractWaveFunction._op_handler(other, self, operator.floordiv)

    def __mod__(
            self,
            other: Union[AbstractWaveFunction, float, int]
    ) -> WaveFunction:
        return AbstractWaveFunction._op_handler(self, other, operator.mod)
    
    def __rmod__(
            self,
            other: Union[AbstractWaveFunction, float, int]
    ) -> WaveFunction:
        return AbstractWaveFunction._op_handler(other, self, operator.mod)

    def __pow__(
            self,
            other: Union[AbstractWaveFunction, float, int]
    ) -> WaveFunction:
        return AbstractWaveFunction._op_handler(self, other, operator.pow)
    
    def __rpow__(
            self,
            other: Union[AbstractWaveFunction, float, int]
    ) -> WaveFunction:
        return AbstractWaveFunction._op_handler(other, self, operator.pow)
  

class WaveFunction(AbstractWaveFunction):
    def __init__(self, function: Expr) -> None:
        # FIXME: assert it's a function of time
        self.function = function


class AbstractFrequencyAmplitudeWave(AbstractWaveFunction):
    def __init__(
            self,
            frequency: Union[Expr, float, int] = Note.A4,
            amplitude: Union[Expr, float, int] = 0.5,
    ) -> None:
        self.frequency = frequency
        self.amplitude = amplitude


class SineWave(AbstractFrequencyAmplitudeWave):
    @property
    def function(self) -> Expr:
        return self.amplitude * sp.sin(2 * sp.pi * self.frequency * TIME)


class SawtoothWave(AbstractFrequencyAmplitudeWave):
    @property
    def function(self) -> Expr:
        return 2 * self.amplitude * (
            time * self.frequency - sp.floor(time * self.frequency + 0.5)
        )
