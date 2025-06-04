from typing import Union

import haiku as hk

import e3nn_jax as e3nn
from jax import nn


class ScaleShiftLayer(hk.Module):
    """Scales and shifts the input by learnable parameters."""

    def __init__(self,
                 scale: Union[float, bool] = True,
                 shift: Union[float, bool] = True):
        super().__init__()

        self.scale = self.get_param("scale", scale, 1.0)
        self.shift = self.get_param("shift", shift, 0.0)

    @staticmethod
    def get_param(name, value, default):
        if isinstance(value, bool):
            if not value:
                return default
            else:
                value = default

        return hk.get_parameter(name, shape=(), init=hk.initializers.Constant(value))

    def __call__(self, x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        return nn.softplus(self.scale) * x + self.shift

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(scale={self.scale:.6f}, shift={self.shift:.6f})"
        )
