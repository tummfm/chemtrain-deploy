from typing import Callable, Optional, Union

import e3nn_jax as e3nn
import flax
import haiku as hk
import jax
import jax.numpy as jnp
from jax import nn

from jax_md_mod.model import layers

from .filter_layers import filter_layers


class InteractionLayer(hk.Module):
    """NequIP interaction (message-passing) layer."""

    def __init__(self,
                 *,
                 node_irreps: e3nn.Irreps,
                 even_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu,
                 odd_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.tanh,
                 avg_num_neighbors: float = 1.0,
                 max_ell: int = 3,
                 n_species: int = 1,
                 mlp_n_layers: int = 2,
                 mlp_n_hidden: int = 64,
                 mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu,
                 ):
        super().__init__(name="Interaction")

        print(f"Node irreps: {node_irreps}")
        self.node_irreps = node_irreps
        self.num_species = n_species

        self.mlp_n_layers = mlp_n_layers
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_activation = mlp_activation

        self.even_activation = even_activation
        self.odd_activation = odd_activation

        self.avg_num_neighbors = avg_num_neighbors
        self.max_ell = max_ell

    def __call__(self,
                 node_feats: e3nn.IrrepsArray,
                 vectors: e3nn.IrrepsArray,
                 rbf: e3nn.IrrepsArray,
                 species: jnp.ndarray,
                 senders: jnp.ndarray,
                 receivers: jnp.ndarray,
                 ) -> e3nn.IrrepsArray:

        num_nodes = node_feats.shape[0]
        num_edges = vectors.shape[0]
        assert vectors.shape == (num_edges, 3)
        assert node_feats.shape == (num_nodes, node_feats.irreps.dim)
        assert species.shape == (num_nodes,)
        assert senders.shape == (num_edges,)
        assert receivers.shape == (num_edges,)

        # we regroup the target irreps to make sure that gate activation
        # has the same irreps as the target
        output_irreps = e3nn.Irreps(self.node_irreps).regroup()

        messages = e3nn.haiku.Linear(node_feats.irreps, name="linear_up")(node_feats)[
            senders]

        # Angular part
        messages = e3nn.concatenate(
            [
                messages.filter(output_irreps + "0e"),
                e3nn.tensor_product(
                    messages,
                    e3nn.spherical_harmonics(
                        [l for l in range(1, self.max_ell + 1)],
                        vectors,
                        normalize=True,
                        normalization="integral",
                    ),
                    filter_ir_out=output_irreps + "0e",
                ),
            ]
        ).regroup()
        assert messages.shape == (num_edges, messages.irreps.dim)

        # Radial part
        with jax.ensure_compile_time_eval():
            assert abs(self.mlp_activation(0.0)) < 1e-6

        mix = e3nn.haiku.MultiLayerPerceptron(
            self.mlp_n_layers * (self.mlp_n_hidden,) + (
            messages.irreps.num_irreps,),
            self.mlp_activation,
            output_activation=False,
        )(rbf)

        # Discard 0 length edges that come from graph padding
        assert mix.shape == (num_edges, messages.irreps.num_irreps)

        # Product of radial and angular part
        messages = messages * mix / node_feats.irreps.num_irreps
        assert messages.shape == (num_edges, messages.irreps.dim)

        # Skip connection. We add scalars for each non-scalar irrep. These
        # scalares will be used to gate the non-scalar irreps.
        irreps = output_irreps.filter(keep=messages.irreps)
        num_nonscalar = irreps.filter(drop="0e + 0o").num_irreps
        irreps = irreps + e3nn.Irreps(f"{2 * num_nonscalar}x0e").simplify()

        # Linear layer learns per-species weights for the skip connection
        skip = e3nn.haiku.Linear(
            irreps,
            num_indexed_weights=self.num_species,
            name="skip_tp",
            force_irreps_out=True,
        )(species, node_feats)

        # Message passing
        node_feats = e3nn.scatter_sum(messages, dst=receivers,
                                      output_size=num_nodes)
        node_feats = node_feats / jnp.sqrt(self.avg_num_neighbors)

        node_feats = e3nn.haiku.Linear(irreps, name="linear_down")(node_feats)

        node_feats = node_feats + skip
        assert node_feats.shape == (num_nodes, node_feats.irreps.dim)

        node_feats = e3nn.gate(
            node_feats,
            even_act=self.even_activation,
            odd_act=self.odd_activation,
            even_gate_act=self.even_activation,
            odd_gate_act=self.odd_activation
        )

        return node_feats



class NequIP(hk.Module):
    """NequIP equivariant GNN for molecular property prediction."""

    def __init__(
        self,
        avg_num_neighbors: float,
        num_species: int = 1,
        max_ell: int = 3,
        embed_dim: int = 32,
        hidden_irreps: e3nn.Irreps = None,
        output_irreps: e3nn.Irreps = None,
        even_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu,
        odd_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.tanh,
        mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu,
        mlp_n_hidden: int = 64,
        mlp_n_layers: int = 2,
        name: Optional[str] = None,
        ):
        super().__init__(name)

        hidden_irreps = [e3nn.Irreps(irreps) for irreps in hidden_irreps]
        output_irreps = [e3nn.Irreps(irreps) for irreps in output_irreps]

        self.species_embedding = hk.Embed(vocab_size=num_species, embed_dim=embed_dim)
        self.rbf = layers.RadialBesselLayer(cutoff=1.0, num_radial=8)

        self.interaction_blocks = [
            InteractionLayer(
                node_irreps=layer_irreps,
                even_activation=even_activation,
                odd_activation=odd_activation,
                avg_num_neighbors=avg_num_neighbors,
                max_ell=max_ell,
                n_species=num_species,
                mlp_activation=mlp_activation,
                mlp_n_hidden=mlp_n_hidden,
                mlp_n_layers=mlp_n_layers
            )
            for layer_irreps in filter_layers(hidden_irreps, max_ell=max_ell)
        ]

        self.output_irreps = output_irreps

        self.num_species = num_species


    def __call__(
        self,
        vectors: e3nn.IrrepsArray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        species: jnp.ndarray,
        mask: jnp.ndarray,
        ):

        # Graph embedding
        node_features = self.species_embedding(species)
        node_features = jnp.where(mask[:, jnp.newaxis], node_features, 0.0)
        node_features = e3nn.as_irreps_array(node_features)

        rbf = self.rbf(e3nn.norm(vectors).array[:, 0])

        # Interaction
        for idx, interaction_block in enumerate(self.interaction_blocks):
            node_features = interaction_block(
                node_feats=node_features,
                vectors=vectors,
                rbf=rbf,
                species=species,
                senders=senders,
                receivers=receivers,
            )


        # Output
        for irreps in self.output_irreps:
            node_features = e3nn.haiku.Linear(irreps)(node_features)

        return node_features
