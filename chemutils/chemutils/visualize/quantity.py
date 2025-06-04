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

"""Create plots of instantaneous quantities."""

import numpy as onp

from .util import create_subplots

__all__ = (
    "plot_adfs",
)

@create_subplots(1, 1)
def plot_adfs(ax, adfs, labels, angles=None, xlabel=r'$\phi$ in rad',
              create_legend: bool = True, reference_first: bool = True):
    """Plot ADFs.

    Args:
        ax: Add to existing axis
        adfs: Adfs to add to the plot
        labels: Label for each adf
        angles: Angles :math:`\\phi` corresponding to the adfs
        xlabel: Label of x-axis
        create_legend: Create a legend with the provided labels
        reference_first: Whether first adf is special, e.g., the reference

    Returns:
        Returns figure and axis of the ADF plot.

    """
    for i, adf in enumerate(adfs):
        # Default angle from 0 to 180 deg.
        if angles is None:
            angles = onp.linspace(0, onp.pi, len(adf))

        if i == 0 and reference_first:
            line = '--'
        else:
            line = '-'

        ax.plot(
            angles, adf, label=labels[i] if labels is not None else None,
            linestyle=line, linewidth=2.0
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel('ADF')

    if create_legend:
        ax.legend()

    return ax

@create_subplots(1, 1)
def plot_rdfs(ax, rdfs, labels, dists=None, xlabel=r'$r$ in nm',
              create_legend: bool = True, reference_first: bool = True):
    """Plot RDFs.

    Args:
        ax: Add to existing axis
        rdfs: Rdfs to add to the plot
        labels: Label for each rdf
        dists: Distances :math:`r` corresponding to the rdfs
        xlabel: Label of x-axis
        create_legend: Create a legend with the provided labels
        reference_first: Whether first rdf is special, e.g., the reference

    Returns:
        Returns figure and axis of the RDF plot.

    """

    # Assume
    if dists is None:
        xlabel=r'$r/R$'

    for i, rdf in enumerate(rdfs):
        if dists is None:
            dists = onp.linspace(0, 1, len(rdf))

        if i == 0 and reference_first:
            line = '--'
        else:
            line = '-'

        ax.plot(
            dists, rdf, label=labels[i] if labels is not None else None,
            linestyle=line, linewidth=2.0
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel('RDF')

    if create_legend:
        ax.legend()

    return ax