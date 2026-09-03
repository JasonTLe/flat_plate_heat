"""Rebunch the mesh's wall-normal spacing to a given first-off-wall cell
height, keeping the node count and domain height fixed.

Why not cgnsutilities' own Block.rebunch(): it's hard-coded to rebunch
along the k-direction starting at the k-min plane (a pyHyp-style mesh
convention, wall-normal = k). This mesh's wall-normal direction is i
(dims = [65 wall-normal, 64 streamwise, 2 spanwise-thickness]; k is the
2-point pseudo-2D extrusion direction, not wall-normal), and the wall is
at i=64 (max), not i=0. This script does the same thing rebunch() does
internally -- fit a spline through each wall-normal line and re-parametrize
it with cgnsutilities.getS()'s geometric-growth-ratio distribution -- just
applied to the correct axis and end for this mesh.

Usage: python3 rebunch_wall_normal.py <inFile> [--spacing S0]

Default --spacing matches the SU2 "Laminar Flat Plate" tutorial's mesh
(su2/incompressible/mesh_flatplate_65x65.su2), whose first off-wall cell is
1.60643705e-05 m (vs. this mesh's original 1.0e-05 m).
"""

import argparse
import os

import numpy as np
from cgnsutilities.cgnsutilities import getS, readGrid
from pyspline import Curve

parser = argparse.ArgumentParser()
parser.add_argument("inFile", type=str)
parser.add_argument("--spacing", type=float, default=1.60643705e-05, help="new first-off-wall cell height, m")
args = parser.parse_args()

base, ext = os.path.splitext(args.inFile)
outFile = f"{base}_rebunch{ext}"

grid = readGrid(args.inFile)

for blk in grid.blocks:
    ni, nj, nk = blk.dims
    newCoords = np.zeros_like(blk.coords)
    for j in range(nj):
        for k in range(nk):
            # Flip so index 0 is the wall (i=ni-1 in the original ordering)
            # and index -1 is the far-field top (i=0) -- what getS()/the
            # curve parametrization assumes (s=0 at the rebunched end).
            xx = blk.coords[::-1, j, k, :]
            c = Curve(X=xx, localInterp=True)

            d = np.linalg.norm(xx[0] - xx[1])  # current first-off-wall spacing
            s0 = (args.spacing / d) * c.s[1]
            newS = getS(ni, s0, c.s[-1])

            newCoords[:, j, k, :] = c(newS)[::-1]  # flip back to i=0..ni-1

    blk.coords = newCoords

grid.writeToCGNS(outFile)
print(f"Rebunched wall-normal spacing to {args.spacing:.6e} m --> {outFile}")
