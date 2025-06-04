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

"""Helper scripts to set up LAMMPS simulations."""

import argparse

import mdtraj

from jax_md_mod import io

__LAMMPS_HEADER = """
# LAMMPS data file written by chemutils
{atoms} atoms
{atomtypes} atom types
0.0 {bx:.4f} xlo xhi
0.0 {by:.4f} ylo yhi
0.0 {bz:.4f} zlo zhi
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, help="Command to run")
    parser.add_argument("infile", type=str, help="Input file")
    parser.add_argument("outfile", type=str, help="Output file")
    parser.add_argument("--scale_pos", type=float, default=10.0, help="Scale factor for positions")
    args = parser.parse_args()

    if args.command == "write_lmpdat":
        write_lmpdat(args.infile, args.outfile, scale_pos=args.scale_pos)
    else:
        raise ValueError(f"Unknown command: {args.command}")


def write_lmpdat(infile, outfile, scale_pos=10.0):

    box, coords, masses, species = io.load_box(infile)

    # Convert to LAMMPS format
    with open(outfile, "w") as f:
        f.write(__LAMMPS_HEADER.format(
            atoms=coords.shape[0],
            atomtypes=species.max(),
            bx=box[0] * scale_pos,
            by=box[1] * scale_pos,
            bz=box[2] * scale_pos
        ))

        f.write("\nMasses\n\n")
        for i in range(1, species.max() + 1):
            elem = mdtraj.element.Element.getByAtomicNumber(i)
            f.write(f"{i} {elem.mass:.4f}\n")

        f.write("\nAtoms\n\n")
        for i, (coord, spec) in enumerate(zip(coords, species)):
            f.write(f"{i + 1} {spec} {coord[0] * scale_pos:.4f} {coord[1] * scale_pos:.4f} {coord[2] * scale_pos:.4f}\n")

if __name__ == "__main__":
    main()
