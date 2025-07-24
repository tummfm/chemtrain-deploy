from copy import deepcopy
from collections import OrderedDict
from functools import partial

import numpy as onp

import jax
import jax.numpy as jnp

import haiku as hk

from jax_md_mod.model import sparse_graph
from jax_md import space, partition, util as md_util, nn

import e3nn_jax

from typing import Tuple, Callable, Any

from .nequip import NequIP
from .nequip_escn import NequIPESCN

from chemutils.models.layers import AtomicEnergyLayer


nequip_default_kwargs = OrderedDict(
    embed_dim = 32,
    input_irreps = "1o",
    output_irreps = ["8x0e", "2x0e"], # Reduce the output trough two linear layers
    hidden_irreps = ["16x0e + 16x1o + 16x1e + 2x2e"] * 4,
    mlp_n_hidden=64,
    mlp_n_layers=2,
    max_ell = 2,
    avg_num_neighbors = 1,
    n_radial_basis = 8
)


def nequip_neighborlist_pp(displacement: space.DisplacementFn,
                           r_cutoff: float,
                           n_species: int = 100,
                           positions_test: jnp.ndarray = None,
                           neighbor_test: partition.NeighborList = None,
                           max_edge_multiplier: float = 1.25,
                           max_edges=None,
                           avg_num_neighbors: float = 1.0,
                           nequip_escn: bool = False,
                           mode: str = "property_prediction",
                           per_particle: bool = False,
                           positive_species: bool = False,
                           pp_network = None,
                           pp_init_args: dict = None,
                           **nequip_kwargs
                           ) -> Tuple[nn.InitFn, Callable[[Any, md_util.Array],
                                                               md_util.Array]]:
    """NequIP model for property prediction.

    Args:
        displacement: Jax_md displacement function
        r_cutoff: Radial cut-off distance of DimeNetPP and the neighbor list
        n_species: Number of different atom species the network is supposed
            to process.
        positions_test: Sample positions to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        neighbor_test: Sample neighborlist to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        max_edge_multiplier: Multiplier for initial estimate of maximum edges.
        max_edges: Expected maximum of valid edges.
        nequip_escn: Use NequIPESCN instead of NequIP (more computational
            efficient).
        avg_num_neighbors: Average number of neighbors per particle. Guessed
            if positions_test and neighbor_test are provided.
        mode: Prediction mode of the model. If "property_prediction" (default),
            returns the learned node features. If "energy_prediction", returns
            the total energy of the system.
        per_particle: Return per-particle energies instead of total energy.
        positive_species: True if the smallest occurring species is 1, e.g., in
            case of atomic numbers.
        nequip_kwargs: Kwargs to change the default structure of NequIP.
            For definition of the kwargs, see NequIP.


    Returns:
        A tuple of 2 functions: A init_fn that initializes the model parameters
        and an energy function that computes the energy for a particular state
        given model parameters. The energy function requires the same input as
        other energy functions with neighbor lists in jax_md.energy.
    """
    kwargs = deepcopy(nequip_default_kwargs)
    kwargs.update(nequip_kwargs)

    r_cutoff = jnp.array(r_cutoff, dtype=md_util.f32)

    # Checking only necessary if neighbor list is dense
    if positions_test is not None and neighbor_test is not None:
        if neighbor_test.format == partition.Dense:
            print('Capping edges and triplets. Beware of overflow, which is'
                  ' currently not being detected.')

            testgraph, _ = sparse_graph.sparse_graph_from_neighborlist(
                displacement, positions_test, neighbor_test, r_cutoff)
            max_edges = jnp.int32(jnp.ceil(testgraph.n_edges * max_edge_multiplier))

            # cap maximum edges and angles to avoid overflow from multiplier
            n_particles, n_neighbors = neighbor_test.idx.shape
            max_edges = min(max_edges, n_particles * n_neighbors)

            print(f"Estimated max. {max_edges} edges.")

            avg_num_neighbors = testgraph.n_edges / n_particles
        else:
            n_particles = neighbor_test.idx.shape[0]
            avg_num_neighbors = onp.sum(neighbor_test.idx[0] < n_particles)
            avg_num_neighbors /= n_particles

    print(f"Average numb neighbors set to {avg_num_neighbors}")

    @hk.without_apply_rng
    @hk.transform
    def model(position: md_util.Array,
              neighbor: partition.NeighborList,
              species: md_util.Array = None,
              mask: md_util.Array = None,
              **dynamic_kwargs):
        if species is None:
            species = jnp.zeros(position.shape[0], dtype=jnp.int32)
        elif positive_species:
            species -= 1
        if mask is None:
            mask = jnp.ones(position.shape[0], dtype=jnp.bool_)

        # Compute the displacements for all edges
        dyn_displacement = partial(displacement, **dynamic_kwargs)

        if neighbor.format == partition.Dense:
            graph, _ = sparse_graph.sparse_graph_from_neighborlist(
                dyn_displacement, position, neighbor, r_cutoff,
                species, max_edges=max_edges, species_mask=mask
            )
            senders = graph.idx_i
            receivers = graph.idx_j
        else:
            assert neighbor.idx.shape == (2, neighbor.idx.shape[1]), "Neighbor list has wrong shape."
            senders, receivers = neighbor.idx

        # Set invalid edges to the cutoff to avoid numerical issues
        vectors = jax.vmap(dyn_displacement)(position[senders], position[receivers])
        vectors = jnp.where(
            jnp.logical_and(
                senders < position.shape[0],
                receivers < position.shape[0]
            )[:, jnp.newaxis], vectors, r_cutoff / jnp.sqrt(3))
        vectors /= r_cutoff

        vectors = e3nn_jax.IrrepsArray(
            e3nn_jax.Irreps(kwargs['input_irreps']), vectors
        )

        if mode == "energy_prediction":
            kwargs["output_irreps"][-1] = "1x0e"
        elif mode == "energy_and_charge":
            kwargs["output_irreps"][-1] = "2x0e"

        net = NequIP(
            avg_num_neighbors=avg_num_neighbors,
            num_species=n_species,
            max_ell=kwargs["max_ell"],
            embed_dim=kwargs["embed_dim"],
            hidden_irreps=kwargs["hidden_irreps"],
            output_irreps=kwargs["output_irreps"],
            mlp_n_hidden=kwargs["mlp_n_hidden"],
            mlp_n_layers=kwargs["mlp_n_layers"],
        )

        features = net(vectors, senders, receivers, species, mask)

        if mode == "energy":
            # Learnable scale and shift
            per_atom_energies, = features.array.T
            per_atom_energies = AtomicEnergyLayer(n_species)(per_atom_energies, species)
            per_atom_energies *= mask

            if per_particle:
                return per_atom_energies
            else:
                return md_util.high_precision_sum(per_atom_energies)

        elif mode == "energy_and_charge":
            # Learnable scale and shift
            charge_scale = hk.get_parameter("charge_scale", shape=(), init=hk.initializers.Constant(1.0))

            per_atom_energies, per_node_charges = features.array.T
            per_atom_energies = AtomicEnergyLayer(n_species)(features.array, species)
            per_atom_energies *= mask

            # Enforce charge neutrality
            per_node_charges = charge_scale * per_node_charges
            per_node_charges -= jnp.sum(mask * per_node_charges) / jnp.sum(mask)
            per_node_charges *= mask

            if per_particle:
                return per_atom_energies, per_node_charges
            else:
                total_energy = md_util.high_precision_sum(per_atom_energies)
                return total_energy, per_node_charges

        elif mode == "property_prediction":
            if pp_network is not None:
                if pp_init_args is None:
                    pp_net = pp_network()
                else:
                    pp_net = pp_network(**pp_init_args)
                print(f"Using property-prediction network")

                return pp_net(features.array, species, mask)

            return features
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

    return jax.jit(model.init), jax.jit(model.apply)
