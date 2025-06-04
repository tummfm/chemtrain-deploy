# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as onp

import jax
from jax import numpy as jnp, nn as jax_nn

import e3nn_jax as e3nn

import haiku as hk

from jax_md_mod.model import layers

from .scale_shift import ScaleShiftLayer

from typing import Callable


class CELLI(hk.Module):

    def __init__(self,
                 charge_embed_n_hidden: int = 16,
                 charge_embed_n_layers: int = 1,
                 num_species: int = 100,
                 mlp_n_hidden: int = 32,
                 mlp_n_layers: int = 3,
                 mlp_activation: Callable = None,
                 envelope_p: int = 6,
                 charge_eq_fn: Callable = None,
                 ):
        super().__init__()

        self.envelope_p = envelope_p

        self.charge_embed_dim = charge_embed_n_hidden
        self.charge_embed_layers = charge_embed_n_layers

        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_activation = mlp_activation
        self.mlp_n_layers = mlp_n_layers

        self.radius = hk.get_parameter(
            "radius", (num_species,), jnp.float32,
            init=hk.initializers.Constant(0.0))
        self.hardness = hk.get_parameter(
            "hardness", (num_species,), jnp.float32,
            init=hk.initializers.Constant(10.0))
        self.chi_shift_scale = ScaleShiftLayer(1.0, False)

        self.charge_embed = hk.get_parameter(
            "charge_embed", (num_species, charge_embed_n_hidden),
            jnp.float32, init=hk.initializers.RandomNormal(1.0)
        )

        self.charge_eq_fn = charge_eq_fn


    def __call__(
        self,
        vectors: e3nn.IrrepsArray,
        x: jnp.ndarray,
        V: e3nn.IrrepsArray,
        senders: jnp.ndarray,
        species: jnp.ndarray,
        ) -> e3nn.IrrepsArray:

        lengths = jnp.sum(vectors.array ** 2, axis=-1) ** 0.5

        # Setup the QEM by computing positional-dependent charges and
        # species-dependent radii and hardnesses
        chis = e3nn.haiku.MultiLayerPerceptron(
            [self.charge_embed_dim] * self.charge_embed_layers + [1,],
            self.mlp_activation,
            output_activation=False
        )(x)
        chis *= layers.SmoothingEnvelope(self.envelope_p)(lengths)[:, None]
        chis = jax.ops.segment_sum(chis, senders, species.size).squeeze(1)
        chis = self.chi_shift_scale(chis)

        gammas = jax_nn.softplus(self.radius[species]) / jnp.log(2)
        hardness = jax_nn.softplus(self.hardness[species])

        # Equilibrate charges
        charges, pot = self.charge_eq_fn(gammas, chis, hardness)

        w = e3nn.haiku.MultiLayerPerceptron(
            [self.charge_embed_dim] * self.charge_embed_layers,
            self.mlp_activation,
            output_activation=False
        )(
            jnp.concatenate((
                charges[:, jnp.newaxis],
                self.charge_embed[species]), axis=-1
            )
        )

        x = e3nn.haiku.MultiLayerPerceptron(
            [self.mlp_n_hidden] * self.mlp_n_layers,
            self.mlp_activation,
            output_activation=False
        )(jnp.concatenate([x, w[senders]], axis=-1))
        x = layers.SmoothingEnvelope(self.envelope_p)(lengths)[:, None] * x

        return x, V, (charges, pot)
