"""
Blasius laminar flat-plate similarity solutions vs SU2 and ADflow CFD results.

Case: SU2 "Laminar Flat Plate" tutorial (lam_flatplate.cfg), run twice:
  compressible/    -> SOLVER= NAVIER_STOKES,      adiabatic wall (no heat transfer)
  incompressible/  -> SOLVER= INC_NAVIER_STOKES,  isothermal cooled wall (148.81 K)

  https://su2code.github.io/tutorials/Laminar_Flat_Plate/

incompressible/ was originally CONSTANT_DENSITY + CONSTANT_VISCOSITY -- with
those, the cooled wall only drove the (one-way-coupled) energy equation, so
the velocity field it produced was mathematically identical to the plain
incompressible Blasius problem regardless of wall temperature (see ADflow's
velocity profile deviating from it below). Now uses VISCOSITY_MODEL=
SUTHERLAND to let the cooled wall actually affect the velocity profile too.
SU2 refuses SUTHERLAND with a truly constant density, though ("Sutherland's
law only valid for ideal gases in incompressible flows"), so
INC_DENSITY_MODEL=VARIABLE / FLUID_MODEL=INC_IDEAL_GAS came along with it --
this run now has both variable viscosity AND variable density, i.e. roughly
ADflow's physics solved via SU2's incompressible (pressure-based) algorithm
instead of ADflow's compressible (density-based) one. See
su2/incompressible/lam_flatplate.cfg for the exact change.

Plus ADflow run on the same freestream/wall spec (adflow_run.py, mesh
meshes/fp_l2_rebunch_fixed_inout.cgns) -- isothermal wall like the
incompressible SU2 case, so it's compared against both the Cf/velocity
theory curves and the incompressible case's Nu(x/L).

Reads (from each SU2 case directory):
  lam_flatplate.cfg   -> freestream conditions (M, T, Re, L)
  surface_flow.vtu    -> wall skin-friction / heat-flux distribution
  flow.vtu             -> volume field, for velocity profiles

Reads (from output/, the ADflow case):
  flat_plate_heat_000_surf.cgns   -> wall SkinFriction, Pressure, Density,
                                      StantonNumber (see NU_ADFLOW below)
  flat_plate_heat_000_vol.cgns    -> volume Velocity field, for the exit
                                      velocity profile

Produces:
  blasius_comparison.png   Cf(x), exit-plane u/U profile, and Nu(x/L)
                            SU2 (compressible + incompressible) + ADflow
                            vs theory
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# ----------------------------------------------------------------------
# 1) Freestream conditions, taken directly from lam_flatplate.cfg
#    (compressible and incompressible cases were set up to match the
#    same Re_L, U_inf, nu_inf, so a single set of reference values is
#    used for both cases' Blasius theory curves)
# ----------------------------------------------------------------------
GAMMA = 1.4
R_GAS = 287.058          # J/(kg K)  -- SU2 default GAS_CONSTANT
MU_REF = 1.716e-5        # kg/(m s)  -- SU2 default Sutherland reference viscosity
T_REF = 273.15           # K
S_SUTH = 110.4           # K

MACH = 0.2
T_INF = 297.62           # K
RE = 1301233.166         # Reynolds number (based on REYNOLDS_LENGTH)
L_REF = 0.3048           # m, REYNOLDS_LENGTH (plate length, x=0..L)

a_inf = np.sqrt(GAMMA * R_GAS * T_INF)
U_INF = MACH * a_inf
MU_INF = MU_REF * (T_INF / T_REF) ** 1.5 * (T_REF + S_SUTH) / (T_INF + S_SUTH)
RHO_INF = RE * MU_INF / (U_INF * L_REF)
NU_INF = MU_INF / RHO_INF

# Incompressible case: variable-viscosity/variable-density INC_IDEAL_GAS
# fluid model (from incompressible/lam_flatplate.cfg) -- see note above
T_WALL_INC = 148.81           # K, MARKER_ISOTHERMAL wall temperature
PRANDTL_LAM = 0.72            # PRANDTL_LAM
CP_INC = 1004.703             # SPECIFIC_HEAT_CP
K_INC = MU_INF * CP_INC / PRANDTL_LAM  # thermal conductivity (W/m-K), CONSTANT_PRANDTL model

print("Freestream properties")
print(f"  a_inf   = {a_inf:.4f} m/s")
print(f"  U_inf   = {U_INF:.4f} m/s")
print(f"  mu_inf  = {MU_INF:.6e} Pa s")
print(f"  rho_inf = {RHO_INF:.6f} kg/m^3")
print(f"  nu_inf  = {NU_INF:.6e} m^2/s")
print(f"  k_inc   = {K_INC:.6e} W/(m K)")

# ----------------------------------------------------------------------
# 2) Blasius similarity solution: f''' + 0.5 f f'' = 0
#    f(0)=0, f'(0)=0, f'(eta_max)=1
# ----------------------------------------------------------------------
def blasius_rhs(eta, y):
    f, fp, fpp = y
    return np.vstack([fp, fpp, -0.5 * f * fpp])

def blasius_bc(ya, yb):
    return np.array([ya[0], ya[1], yb[1] - 1.0])

eta_max = 10.0
eta = np.linspace(0, eta_max, 2000)
y_guess = np.zeros((3, eta.size))
y_guess[1] = eta / eta_max  # rough guess for f'

sol = solve_bvp(blasius_rhs, blasius_bc, eta, y_guess, tol=1e-10, max_nodes=100000)
if not sol.success:
    raise RuntimeError("Blasius BVP failed to converge: " + sol.message)

f_of_eta = sol.sol
FPP0 = sol.sol(0)[2]
print(f"\nBlasius f''(0) = {FPP0:.6f}  (reference value 0.33206)")

def blasius_velocity(eta_query):
    return f_of_eta(eta_query)[1]

# ----------------------------------------------------------------------
# 3) VTK helpers
# ----------------------------------------------------------------------
def read_vtu(fname):
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(fname)
    r.Update()
    return r.GetOutput()

def probe_column(vol, x_val, y_max, n=200):
    line = vtk.vtkLineSource()
    line.SetResolution(n - 1)
    line.SetPoint1(x_val, 0.0, 0.0)
    line.SetPoint2(x_val, y_max, 0.0)
    line.Update()

    probe = vtk.vtkProbeFilter()
    probe.SetInputConnection(line.GetOutputPort())
    probe.SetSourceData(vol)
    probe.Update()
    out = probe.GetOutput()

    pts = vtk_to_numpy(out.GetPoints().GetData())
    vel = vtk_to_numpy(out.GetPointData().GetArray("Velocity"))
    valid = vtk_to_numpy(out.GetPointData().GetArray("vtkValidPointMask")).astype(bool)
    return pts[valid, 1], vel[valid, 0]

CASES = {
    # incompressible is drawn first as a larger hollow ring, compressible on
    # top as a smaller filled dot -- keeps both visible where points coincide
    "incompressible": dict(color="blue", label="SU2 Incompressible (Sutherland)",
                            ms=8, mfc="none", mec="blue", mew=1.4, zorder=3),
    "compressible": dict(color="red", label="SU2 Compressible",
                          ms=4, mfc="red", mec="white", mew=0.6, zorder=4),
}

for name, case in CASES.items():
    case["wall"] = read_vtu(f"su2/{name}/surface_flow.vtu")
    case["vol"] = read_vtu(f"su2/{name}/flow.vtu")

# ----------------------------------------------------------------------
# 3b) ADflow case (output/) -- isothermal wall at the same T_WALL_INC,
#     read via VTK's CGNS reader instead of the .vtu reader used for SU2.
# ----------------------------------------------------------------------
ADFLOW_SURF = "output/flat_plate_heat_000_surf.cgns"
ADFLOW_VOL = "output/flat_plate_heat_000_vol.cgns"
ADFLOW_STYLE = dict(color="purple", label="ADflow", ms=5, mfc="purple", mec="white", mew=0.6, alpha=0.9, zorder=5)

def read_cgns(fname):
    rdr = vtk.vtkCGNSReader()
    rdr.SetFileName(fname)
    rdr.UpdateInformation()
    rdr.EnableAllBases()
    rdr.EnableAllFamilies()
    rdr.EnableAllCellArrays()
    rdr.EnableAllPointArrays()
    rdr.Update()
    return rdr.GetOutput()

def cgns_zone_by_name(multiblock, needle):
    base = multiblock.GetBlock(0)
    for i in range(base.GetNumberOfBlocks()):
        md = base.GetMetaData(i)
        name = md.Get(vtk.vtkCompositeDataSet.NAME()) if md else ""
        if needle in name:
            return base.GetBlock(i)
    raise KeyError(f"no CGNS zone matching '{needle}' in {[base.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME()) for i in range(base.GetNumberOfBlocks())]}")

adflow_wall = cgns_zone_by_name(read_cgns(ADFLOW_SURF), "Isothermal")

cc = vtk.vtkCellCenters()
cc.SetInputData(adflow_wall)
cc.Update()
adflow_x = vtk_to_numpy(cc.GetOutput().GetPoints().GetData())[:, 0]
adflow_order = np.argsort(adflow_x)
adflow_x = adflow_x[adflow_order]
adflow_mask = adflow_x > 1e-6

adflow_wall_cd = adflow_wall.GetCellData()
adflow_cf = vtk_to_numpy(adflow_wall_cd.GetArray("SkinFriction"))[adflow_order, 0]
# StantonNumber, Pressure and Density are all in ADflow's internal
# nondimensional units (freestream pInf = rhoInf = 1 by construction --
# ADflow's reference pressure/density *are* the dimensional freestream
# static P, rho), which is what the wall heat flux is recovered from
# below (see NU_ADFLOW).
adflow_ch = vtk_to_numpy(adflow_wall_cd.GetArray("StantonNumber"))[adflow_order]
adflow_P_nd = vtk_to_numpy(adflow_wall_cd.GetArray("Pressure"))[adflow_order]
adflow_rho_nd = vtk_to_numpy(adflow_wall_cd.GetArray("Density"))[adflow_order]

adflow_vol = cgns_zone_by_name(read_cgns(ADFLOW_VOL), "blk")
adflow_vol_c2p = vtk.vtkCellDataToPointData()
adflow_vol_c2p.SetInputData(adflow_vol)
adflow_vol_c2p.Update()
adflow_vol_pd = adflow_vol_c2p.GetOutput()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

# ----------------------------------------------------------------------
# 4) Skin friction: SU2 (both cases) vs Blasius Cf(x) = 0.664/sqrt(Re_x)
# ----------------------------------------------------------------------
theory_plotted = False
for name, case in CASES.items():
    wall_pts = vtk_to_numpy(case["wall"].GetPoints().GetData())
    cf_wall = vtk_to_numpy(case["wall"].GetPointData().GetArray("Skin_Friction_Coefficient"))

    order = np.argsort(wall_pts[:, 0])
    x_wall = wall_pts[order, 0]
    cf_x = cf_wall[order, 0]

    mask = x_wall > 1e-6  # skip leading edge singularity
    x_wall_m = x_wall[mask]
    cf_su2 = cf_x[mask]
    case["x_wall"] = x_wall_m
    case["cf_su2"] = cf_su2

    if not theory_plotted:
        Re_x = U_INF * x_wall_m / NU_INF
        cf_blasius = 0.664 / np.sqrt(Re_x)
        ax1.plot(
            x_wall_m, cf_blasius, "-",
            color="black", lw=1.5, zorder=2,
            label=r"Blasius: $0.664/\sqrt{Re_x}$",
        )
        theory_plotted = True

    ax1.plot(
        x_wall_m, cf_su2, "o",
        ms=case["ms"], mfc=case["mfc"], mec=case["mec"], mew=case["mew"],
        alpha=0.9, zorder=case["zorder"], label=case["label"],
    )

ax1.plot(
    adflow_x[adflow_mask], adflow_cf[adflow_mask], "o",
    ms=ADFLOW_STYLE["ms"], mfc=ADFLOW_STYLE["mfc"], mec=ADFLOW_STYLE["mec"], mew=ADFLOW_STYLE["mew"],
    alpha=ADFLOW_STYLE["alpha"], zorder=ADFLOW_STYLE["zorder"], label=ADFLOW_STYLE["label"],
)

ax1.set_xlabel("x (m)")
ax1.set_ylabel(r"$C_f$")
ax1.set_xlim(-0.03, 0.35)
ax1.set_ylim(0.000, 0.008)
ax1.set_xticks(np.arange(0.00, 0.35 + 1e-9, 0.05))
ax1.set_yticks(np.arange(0.000, 0.008 + 1e-9, 0.001))
ax1.set_title("Skin friction")
ax1.legend()
ax1.grid(alpha=0.3)

# ----------------------------------------------------------------------
# 5) Velocity profiles u/U_inf vs eta at the exit plane (x = L), both cases
# ----------------------------------------------------------------------
x_val = L_REF  # exit plane (trailing edge, x = L)

delta99_guess = 5.0 * x_val / np.sqrt(U_INF * x_val / NU_INF)  # ~Blasius delta99
y_max = max(3.0 * delta99_guess, 0.002)

eta_ref = np.linspace(0, 9, 200)
ax2.plot(
    blasius_velocity(eta_ref), eta_ref, "-",
    color="black", lw=1.5, zorder=2, label="Blasius",
)

def thin_by_eta(eta_arr, n=25, eta_max=9.0):
    """Pick ~n marker indices spread evenly across [0, eta_max] *in eta*,
    instead of uniform index spacing along the (linearly y-spaced) probe.
    The boundary layer only occupies the first ~third of the probe line --
    uniform index spacing wastes most of its points on the flat free-stream
    plateau beyond it, piling dozens of overlapping open circles on top of
    each other there. Evenly spacing in eta instead gives each marker its
    own visual slot the whole way up."""
    targets = np.linspace(0, eta_max, n)
    idx = np.searchsorted(eta_arr, targets)
    idx = np.clip(idx, 0, len(eta_arr) - 1)
    return np.unique(idx)

for name, case in CASES.items():
    y, u_raw = probe_column(case["vol"], x_val, y_max, n=200)
    # compressible Velocity field is dimensional (m/s); incompressible
    # Velocity field is already non-dimensionalized by U_inf
    u_over_U = u_raw / U_INF if name == "compressible" else u_raw
    eta_probe = y * np.sqrt(U_INF / (NU_INF * x_val))
    idx = thin_by_eta(eta_probe)
    ax2.plot(
        u_over_U[idx], eta_probe[idx], "o",
        ms=case["ms"], mfc=case["mfc"], mec=case["mec"], mew=case["mew"],
        alpha=0.9, zorder=case["zorder"], label=f"{case['label']} (exit plane, x=L)",
    )

# ADflow: Velocity is nondimensionalized by sqrt(pRef/rhoRef), not U_inf,
# so u/U_inf = u_nondim / (Mach * sqrt(gamma)) rather than a raw copy.
y_adflow, u_adflow_nd = probe_column(adflow_vol_pd, x_val - 1e-4, y_max)
u_over_U_adflow = u_adflow_nd / (MACH * np.sqrt(GAMMA))
eta_adflow = y_adflow * np.sqrt(U_INF / (NU_INF * x_val))
idx_adflow = thin_by_eta(eta_adflow)
ax2.plot(
    u_over_U_adflow[idx_adflow], eta_adflow[idx_adflow], "o",
    ms=ADFLOW_STYLE["ms"], mfc=ADFLOW_STYLE["mfc"], mec=ADFLOW_STYLE["mec"], mew=ADFLOW_STYLE["mew"],
    alpha=ADFLOW_STYLE["alpha"], zorder=ADFLOW_STYLE["zorder"],
    label=f"{ADFLOW_STYLE['label']} (exit plane, x=L)",
)

ax2.set_xlabel(r"$u / U_\infty$")
ax2.set_ylabel(r"$\eta = y\sqrt{U_\infty/(\nu x)}$")
ax2.set_xlim(0.0, 1.1)
ax2.set_ylim(0, 9)
ax2.set_xticks(np.arange(0.0, 1.1 + 1e-9, 0.2))
ax2.set_yticks(np.arange(0, 9 + 1e-9, 1))
ax2.set_title("Velocity profile (exit plane, x=L)")
ax2.legend(loc="upper left", framealpha=1.0)
ax2.grid(alpha=0.3)

# ----------------------------------------------------------------------
# 6) Nusselt number vs x/L (incompressible only -- the compressible case
#    runs an adiabatic wall, MARKER_HEATFLUX=0, so it has no wall heat
#    transfer to compare). Nu_x = q_wall * x / (k * (T_inf - T_wall)).
# ----------------------------------------------------------------------
inc = CASES["incompressible"]
wall_pts = vtk_to_numpy(inc["wall"].GetPoints().GetData())
q_wall = vtk_to_numpy(inc["wall"].GetPointData().GetArray("Heat_Flux"))
T_wall_field = vtk_to_numpy(inc["wall"].GetPointData().GetArray("Temperature"))  # normalized by T_inf

order = np.argsort(wall_pts[:, 0])
x_wall = wall_pts[order, 0]
q_wall = q_wall[order]
T_wall_field = T_wall_field[order]

mask = x_wall > 1e-6
x_wall_m = x_wall[mask]
q_wall_m = q_wall[mask]
dT = T_INF - T_WALL_INC  # K

Nu_su2 = q_wall_m * x_wall_m / (K_INC * dT)
xoL_su2 = x_wall_m / L_REF

xoL_theory = np.linspace(1e-4, 1.0, 200)
Re_x_theory = U_INF * (xoL_theory * L_REF) / NU_INF
Nu_theory = 0.332 * np.sqrt(Re_x_theory) * PRANDTL_LAM ** (1.0 / 3.0)

ax3.plot(
    xoL_theory, Nu_theory, "-",
    color="black", lw=1.5, zorder=2,
    label=r"Blasius: $0.332\sqrt{Re_x}\,Pr^{1/3}$",
)
ax3.plot(
    xoL_su2, Nu_su2, "o",
    ms=4, mfc=inc["color"], mec="white", mew=0.6, alpha=0.9,
    zorder=3, label=inc["label"],
)

# ---------------------------------------------------------------------
# ADflow Nu(x): the CGNS surface output has no raw heat-flux field --
# ADflow's only surface encoding of it is "ch" (Stanton number), written
# using the compressible/adiabatic-wall-referenced formula from
# outputMod.F90 (case cgnsStanton):
#   qw = ch * fact * (a2Tot - a2),  fact = MachCoef*sqrt(gamma*pInf*rhoInf)/(gamma-1)
#   a2Tot = gamma*pInf*(1 + 0.5*(gamma-1)*MachCoef^2)/rhoInf,  a2 = gamma*P/rho
# evaluated in ADflow's own nondimensional unit system, where pInf =
# rhoInf = 1 by construction (ADflow's pRef/rhoRef *are* the dimensional
# freestream static P/rho, so the wall's Pressure/Density CGNS fields are
# already expressed in exactly these units). Redimensionalize with
# ADflow's own heat-flux scale, pRef*sqrt(pRef/rhoRef) (see
# heatFluxes() in BCRoutines.F90), using pRef = P_INF, rhoRef = RHO_INF.
# ---------------------------------------------------------------------
gm1 = GAMMA - 1.0
a2Tot_nd = GAMMA * (1.0 + 0.5 * gm1 * MACH**2)  # pInf_nd = rhoInf_nd = 1
fact_nd = MACH * np.sqrt(GAMMA) / gm1
a2_nd = GAMMA * adflow_P_nd / adflow_rho_nd
qw_nd = adflow_ch * fact_nd * (a2Tot_nd - a2_nd)

P_INF = RHO_INF * R_GAS * T_INF  # ideal gas static freestream pressure == ADflow's pRef
scaleDim = P_INF * np.sqrt(P_INF / RHO_INF)
qw_adflow = qw_nd * scaleDim  # W/m^2

Nu_adflow = qw_adflow[adflow_mask] * adflow_x[adflow_mask] / (K_INC * dT)
xoL_adflow = adflow_x[adflow_mask] / L_REF

ax3.plot(
    xoL_adflow, Nu_adflow, "o",
    ms=ADFLOW_STYLE["ms"], mfc=ADFLOW_STYLE["mfc"], mec=ADFLOW_STYLE["mec"], mew=ADFLOW_STYLE["mew"],
    alpha=ADFLOW_STYLE["alpha"], zorder=ADFLOW_STYLE["zorder"], label=ADFLOW_STYLE["label"],
)

ax3.set_xlabel(r"$x/L$")
ax3.set_ylabel(r"$Nu_x$")
ax3.set_xlim(-0.1, 1.1)
ax3.set_ylim(0, 350)
ax3.set_xticks(np.arange(-0.1, 1.1 + 1e-9, 0.2))
ax3.set_yticks(np.arange(0, 350 + 1e-9, 50))
ax3.set_title("Nusselt number")
ax3.legend()
ax3.grid(alpha=0.3)

fig.suptitle("SU2 (lam_flatplate) + ADflow vs Blasius similarity solutions")
fig.tight_layout()
fig.savefig("blasius_comparison.png", dpi=150)
print("Wrote blasius_comparison.png")
