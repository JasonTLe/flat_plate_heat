"""
Laminar flat-plate case, matching the free-stream/BC spec given for this run:

    Density (constant)          = 1.13235   kg/m^3
    Inlet velocity magnitude     = 69.1687   m/s
    Inlet flow direction (x,y,z) = (1.0, 0.0, 0.0)
    Inlet temperature            = 297.62    K
    Outlet pressure               = 0.0       N/m^2   (see note below)
    Viscosity (constant)          = 1.83463e-05 kg/(m-s)
    Prandtl number (constant)     = 0.72
    Steady, 2D, laminar, incompressible Navier-Stokes
    Multigrid
    Flux-difference splitting (FDS), 2nd-order upwind
    Euler implicit time integration
    Inlet / outlet / symmetry / isothermal no-slip wall BCs, from meshes/

These are exactly the free-stream conditions of the SU2 "Laminar Flat
Plate" tutorial (INC_NAVIER_STOKES branch, see su2/blasius_compare.py):
mach 0.2 at T=297.62 K gives this V, and standard-air Sutherland's law at
that T gives this mu -- so ADflow's default gas/Sutherland constants
reproduce both exactly when driven by (rho, T, V) below.

Notes on ADflow vs. the spec above:
  - ADflow is a compressible solver; it has no incompressible/constant-
    viscosity mode. Viscosity therefore still follows Sutherland's law,
    but ADflow's default Sutherland constants reproduce the requested
    1.83463e-05 kg/(m-s) at T=297.62 K to within 0.02%, so it is
    "constant" in the sense that matters here (freestream state).
  - "Outlet pressure = 0" is the incompressible solver's gauge-pressure
    convention (reference pressure = 0). A compressible solver needs an
    absolute static pressure, so P is instead derived from the ideal gas
    law P = rho*R*T (~96740 Pa), which is what AeroProblem(rho, T, V)
    below computes automatically and applies at the outflow boundary.
  - "Multigrid": meshes/fp_l2_fixed.cgns is 65x64x2 nodes -> 64x63x1
    cells. 63 is odd, so no direction can be halved even once -- this
    mesh does not support multigrid at all, hence MGCycle="sg" (single
    grid, same as the option's own default). --mgcycle is exposed below
    in case a multigrid-compatible mesh (node counts of 2^k+1) is used.
  - "Euler implicit time integration": ADflow's nearest equivalent for a
    steady solve is the ANK solver, which advances the state with an
    implicitly-integrated (backward Euler) pseudo-transient term each
    nonlinear iteration; NK takes over for quadratic convergence once
    the residual is low enough. Both are enabled below.
  - "FDS, 2nd-order upwind": ADflow's discretization="upwind" is Roe's
    flux-difference-splitting scheme, with MUSCL + a limiter
    (limiter="van Albada", the default) giving 2nd-order reconstruction.
  - The isothermal wall temperature (148.81 K) is baked into the mesh by
    fix_bc.py (family "wall_viscous_iso"), since ADflow reads it at
    grid-load time. --Twall lets you override it via AeroProblem.setBCVar()
    for a parameter sweep.

"""
import numpy as np
import argparse
import os
from adflow import ADFLOW
from baseclasses import AeroProblem
from mpi4py import MPI

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="output")
parser.add_argument("--gridFile", type=str, default="./meshes/fp_l0_rebunch_fixed_inout.cgns")
parser.add_argument("--mgcycle", type=str, default="sg")
parser.add_argument("--Twall", type=float, default=None, help="override the wall temperature (K) baked into the mesh")
args = parser.parse_args()

# MPI communicator
comm = MPI.COMM_WORLD

# Create output directory
if comm.rank == 0 and not os.path.exists(args.output):
    os.makedirs(args.output)

# -------------------
# ADflow Solver Setup
# -------------------
aeroOptions = {
    "gridFile": args.gridFile,
    "outputDirectory": args.output,
    "monitorvariables": ["resrho", "resmom", "resrhoe", "yplus"],
    "writeTecplotSurfaceSolution": True,
    # The CGNS surface writer needs a parallel-HDF5 build; ours isn't one, and
    # in parallel it segfaults on whichever rank owns few/no wall faces after
    # partitioning. The Tecplot surface file above uses a separate, non-HDF5
    # writer and carries the same surfaceVariables, so just skip CGNS here.
    "writeSurfaceSolution": True,
    "surfaceVariables": ["cf", "cfx", "cfy", "cfz", "p", "vx", "vy", "vz", "temp", "rho", "mach", "yplus", "ch"],
    "equationType": "laminar NS",
    "equationMode": "steady",
    "discretization": "upwind",       # Roe FDS
    "coarseDiscretization": "upwind",
    "limiter": "van Albada",          # 2nd-order MUSCL reconstruction
    "smoother": "DADI",
    "MGCycle": args.mgcycle,          # "sg": this mesh has no multigrid-compatible dims
    "useANKSolver": True,             # implicit (backward-Euler-style) nonlinear solver
    "useNKSolver": True,
    "CFL": 1.0,
    "L2Convergence": 1e-12,
    "nCycles": 5000,
}

CFDSolver = ADFLOW(options=aeroOptions, comm=comm)

# ------------------------
# AeroProblem Configuration
# ------------------------
ap = AeroProblem(
    name="flat_plate_heat",
    V=69.1687,       # inlet velocity magnitude, m/s
    T=297.62,        # inlet (freestream) static temperature, K
    rho=1.13235,     # constant freestream density, kg/m^3
    alpha=0.0,       # flow direction (1, 0, 0): purely along +x, no incidence
    beta=0.0,
    areaRef=1.0,
    chordRef=1.0,
    evalFuncs=["cd", "cl", "cdv", "cdp"],
)

if args.Twall is not None:
    ap.setBCVar("Temperature", args.Twall, "wall_viscous_iso")

# --------------------
# Run ADflow & Evaluate
# --------------------
CFDSolver(ap)

funcs = {}
CFDSolver.evalFunctions(ap, funcs)

# Print output
if comm.rank == 0:
    print("\n ADflow simulation complete!")
    for key, val in funcs.items():
        print(f"{key:20s} : {val:.6e}")
