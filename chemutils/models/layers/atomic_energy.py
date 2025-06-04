import haiku as hk

import jax.numpy as jnp

from . import ScaleShiftLayer


class AtomicEnergyLayer(hk.Module):
    """Adds atomic energy to the model."""

    def __init__(self,
                 num_species: int,
                 scale: float = 1.0,
                 shift: float = 0.0,
                 species_shift: bool=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.species_shift = species_shift
        self.scale_shift = ScaleShiftLayer(scale=scale, shift=shift)
        self.atomic_energies = hk.Embed(
            num_species, 1,
            embedding_matrix=jnp.zeros((num_species, 1)),
        )

    def __call__(self, per_atom_energies, species):
        if self.species_shift:
            atomic_energies = self.atomic_energies(species).squeeze(axis=1)
        else:
            atomic_energies = jnp.zeros_like(species, dtype=jnp.float32)
        return atomic_energies + self.scale_shift(per_atom_energies)
