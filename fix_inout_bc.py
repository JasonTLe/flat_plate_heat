"""Replace the generic characteristic bcinflow/bcoutflow boundaries with proper
bcinflowsubsonic/bcoutflowsubsonic boundaries carrying the SU2 "Laminar Flat
Plate" tutorial's inlet/outlet spec as baked-in Dirichlet BCData.

Why: ADflow's CGNS reader (readCGNSGrid.F90) maps the *generic* CGNS types
BCInflow and BCOutflow to its internal 'Farfield' (characteristic/Riemann)
treatment -- the same weak, open-boundary treatment used for a true far-field
boundary. That's fine for the top boundary of this domain (which really is a
far-field), but the streamwise inlet/outlet faces of a bounded duct-like flat
plate domain are not a far-field: the tutorial specifies a hard velocity/
temperature inlet and a hard static-pressure outlet, not a Riemann-invariant
boundary. Driving both ends of a thin, strongly wall-cooled channel with weak
characteristic BCs lets the boundary state drift as the ANK CFL ramps up,
which is why the previous mesh (bcinflow/bcoutflow) stalled/oscillated
instead of converging.

ADflow's true subsonic-inflow BC only accepts total conditions + a flow
direction (BCData.F90:BCDataSubsonicInflow -- the density+velocity branch is
unimplemented), so the tutorial's static (rho, T, V) at M=0.2 is converted
to (Ptot, Ttot) via the isentropic relations and baked in along with a unit
flow-direction vector. The outlet just needs the static pressure implied by
the same freestream state (ideal gas law), matching the compressible
equivalent of the tutorial's "outlet pressure = 0 (gauge)".

Usage: python3 fix_inout_bc.py <inFile> [--rho RHO] [--T T] [--V V]
"""

import argparse
import ctypes
import os

import numpy as np
from cgnsutilities.cgnsutilities import BocoDataSet, BocoDataSetArray, CGNSDATATYPES


def setSIUnits(fileName):
    """Stamp DataClass=Dimensional + DimensionalUnits=SI on every BCData_t node.

    Copied from fix_bc.py (which is a script, not importable as a module) --
    see that file's docstring for why this is needed.
    """
    lib = ctypes.CDLL("libcgns.so")
    lib.cg_get_error.restype = ctypes.c_char_p

    CG_MODE_MODIFY, CG_OK = 2, 0
    Dimensional = 2
    Kilogram, Meter, Second, Kelvin, Radian = 2, 2, 2, 2, 3
    Dirichlet, Neumann = 2, 3

    def chk(ierr, what):
        if ierr != CG_OK:
            raise RuntimeError(f"{what} failed: {lib.cg_get_error().decode()}")

    def goto(fn, B, Z, BC, DS, dirNeu):
        labels = [b"Zone_t", b"ZoneBC_t", b"BC_t", b"BCDataSet_t", b"BCData_t"]
        nums = [Z, 1, BC, DS, dirNeu]
        return lib.cg_golist(
            fn,
            B,
            len(labels),
            (ctypes.c_char_p * len(labels))(*labels),
            (ctypes.c_int * len(nums))(*nums),
        )

    fnRef = ctypes.c_int()
    chk(lib.cg_open(fileName.encode(), CG_MODE_MODIFY, ctypes.byref(fnRef)), "cg_open")
    fn = fnRef.value

    nStamped = 0
    try:
        nBases = ctypes.c_int()
        chk(lib.cg_nbases(fn, ctypes.byref(nBases)), "cg_nbases")
        for B in range(1, nBases.value + 1):
            nZones = ctypes.c_int()
            chk(lib.cg_nzones(fn, B, ctypes.byref(nZones)), "cg_nzones")
            for Z in range(1, nZones.value + 1):
                nBocos = ctypes.c_int()
                chk(lib.cg_nbocos(fn, B, Z, ctypes.byref(nBocos)), "cg_nbocos")
                for BC in range(1, nBocos.value + 1):
                    DS = 1
                    while True:
                        hit = False
                        for dirNeu in (Dirichlet, Neumann):
                            if goto(fn, B, Z, BC, DS, dirNeu) != CG_OK:
                                continue
                            hit = True
                            chk(lib.cg_dataclass_write(Dimensional), "cg_dataclass_write")
                            chk(
                                lib.cg_units_write(Kilogram, Meter, Second, Kelvin, Radian),
                                "cg_units_write",
                            )
                            nStamped += 1
                        if not hit:
                            break
                        DS += 1
    finally:
        lib.cg_close(fn)

    return nStamped

GAMMA = 1.4
RGAS = 287.0549254782546  # matches baseclasses.AeroProblem's default R for air

parser = argparse.ArgumentParser()
parser.add_argument("inFile", type=str)
parser.add_argument("--rho", type=float, default=1.13235, help="freestream density, kg/m^3")
parser.add_argument("--T", type=float, default=297.62, help="freestream static temperature, K")
parser.add_argument("--V", type=float, default=69.1687, help="freestream velocity magnitude, m/s")
args = parser.parse_args()

base, ext = os.path.splitext(args.inFile)
outFile = f"{base}_inout{ext}"

# Isentropic total conditions from the tutorial's static freestream state.
Cp = GAMMA * RGAS / (GAMMA - 1)
P = args.rho * RGAS * args.T
Ttot = args.T + args.V**2 / (2 * Cp)
Ptot = P * (Ttot / args.T) ** (GAMMA / (GAMMA - 1))

print(f"Static P    = {P:.6f} Pa")
print(f"Ptot inlet  = {Ptot:.6f} Pa")
print(f"Ttot inlet  = {Ttot:.6f} K")
print(f"P outlet    = {P:.6f} Pa (static)")


def scalarArray(name, value):
    dataArr = np.array([value], dtype=np.float64)
    dataDims = np.ones(3, dtype=np.int32, order="F")
    dataDims[0] = 1
    return BocoDataSetArray(name, CGNSDATATYPES["RealDouble"], 1, dataDims, dataArr)


from cgnsutilities.cgnsutilities import readGrid  # noqa: E402  (after fix_bc import, avoids cycle)

grid = readGrid(args.inFile)

nInflow = 0
nOutflow = 0
for blk in grid.blocks:
    for boco in blk.bocos:
        if boco.internalType == "bcinflow":
            boco.internalType = "bcinflowsubsonic"
            boco.setBCType(boco.internalType)
            boco.family = "inflow_subsonic"

            ds = BocoDataSet("BCDataSet_1", "bcinflowsubsonic")
            ds.addDirichletDataSet(scalarArray("PressureStagnation", Ptot))
            ds.addDirichletDataSet(scalarArray("TemperatureStagnation", Ttot))
            ds.addDirichletDataSet(scalarArray("VelocityUnitVectorX", 1.0))
            ds.addDirichletDataSet(scalarArray("VelocityUnitVectorY", 0.0))
            ds.addDirichletDataSet(scalarArray("VelocityUnitVectorZ", 0.0))
            boco.addBocoDataSet(ds)
            nInflow += 1

        elif boco.internalType == "bcoutflow":
            boco.internalType = "bcoutflowsubsonic"
            boco.setBCType(boco.internalType)
            boco.family = "outflow_subsonic"

            ds = BocoDataSet("BCDataSet_1", "bcoutflowsubsonic")
            ds.addDirichletDataSet(scalarArray("Pressure", P))
            boco.addBocoDataSet(ds)
            nOutflow += 1

grid.writeToCGNS(outFile)
nUnits = setSIUnits(outFile)
print("=" * 50)
print(f"Tagged {nUnits} BCData node(s) with DataClass=Dimensional, units = SI")
print(f"Converted {nInflow} bcinflow -> bcinflowsubsonic, {nOutflow} bcoutflow -> bcoutflowsubsonic --> {outFile}")
