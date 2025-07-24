from typing import Callable, Optional, Union

import e3nn_jax as e3nn
import flax
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax.experimental.linear_shtp import LinearSHTP

from .radial import default_radial_basis

class NequIPESCN(hk.Module):
    def __init__(
        self,
        avg_num_neighbors: float,
        num_species: int = 1,
        output_irreps: e3nn.Irreps = 16 * e3nn.Irreps("0e + 1o + 2e"),
        even_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu,
        odd_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.tanh,
        gate_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu,
        mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu,
        mlp_n_hidden: int = 64,
        mlp_n_layers: int = 2,
        radial_basis: Callable[[jnp.ndarray, int], jnp.ndarray] = default_radial_basis,
        n_radial_basis: int = 8,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.avg_num_neighbors = avg_num_neighbors
        self.num_species = num_species
        self.output_irreps = output_irreps
        self.even_activation = even_activation
        self.odd_activation = odd_activation
        self.gate_activation = gate_activation
        self.mlp_activation = mlp_activation
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_n_layers = mlp_n_layers
        self.radial_basis = radial_basis
        self.n_radial_basis = n_radial_basis

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,
        node_feats: e3nn.IrrepsArray,
        node_species: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ):

        num_nodes = node_feats.shape[0]
        num_edges = vectors.shape[0]
        assert vectors.shape == (num_edges, 3)
        assert node_feats.shape == (num_nodes, node_feats.irreps.dim)
        assert senders.shape == (num_edges,)
        assert receivers.shape == (num_edges,)

        # we regroup the target irreps to make sure that gate activation
        # has the same irreps as the target
        output_irreps = e3nn.Irreps(self.output_irreps).regroup()

        messages = e3nn.haiku.Linear(node_feats.irreps, name="linear_up")(node_feats)[senders]

        conv = LinearSHTP(output_irreps, mix=False)
        w_unused = conv.init(jax.random.PRNGKey(0), messages[0], vectors[0])
        w_unused_flat = flatten(w_unused)

        # Radial part
        with jax.ensure_compile_time_eval():
            assert abs(self.mlp_activation(0.0)) < 1e-6
        lengths = e3nn.norm(vectors).array
        mix = e3nn.haiku.MultiLayerPerceptron(
            self.mlp_n_layers * (self.mlp_n_hidden,) + (w_unused_flat.size,),
            self.mlp_activation,
            output_activation=False,
        )(self.radial_basis(lengths[:, 0], self.n_radial_basis))

        # Discard 0 length edges that come from graph padding
        mix = jnp.where(lengths == 0.0, 0.0, mix)
        assert mix.shape == (num_edges, w_unused_flat.size)

        w = jax.vmap(unflatten, (0, None))(mix, w_unused)
        messages = jax.vmap(conv.apply)(w, messages, vectors)
        assert messages.shape == (num_edges, messages.irreps.dim)

        # Skip connection
        irreps = output_irreps.filter(keep=messages.irreps)
        num_nonscalar = irreps.filter(drop="0e + 0o").num_irreps
        irreps = irreps + e3nn.Irreps(f"{num_nonscalar}x0e").simplify()

        skip = e3nn.haiku.Linear(
            irreps,
            num_indexed_weights=self.num_species,
            name="skip_tp",
            force_irreps_out=True,
        )(node_species, node_feats)

        # Message passing
        node_feats = e3nn.scatter_sum(messages, dst=receivers, output_size=num_nodes)
        node_feats = node_feats / jnp.sqrt(self.avg_num_neighbors)

        node_feats = e3nn.haiku.Linear(irreps, name="linear_down")(node_feats)

        node_feats = node_feats + skip
        assert node_feats.shape == (num_nodes, node_feats.irreps.dim)

        node_feats = e3nn.gate(
            node_feats,
            even_act=self.even_activation,
            odd_act=self.odd_activation,
            even_gate_act=self.gate_activation,
        )

        return node_feats


def flatten(w):
    return jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(w)])


def unflatten(array, template):
    lst = []
    start = 0
    for x in jax.tree_util.tree_leaves(template):
        lst.append(array[start : start + x.size].reshape(x.shape))
        start += x.size
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(template), lst)
