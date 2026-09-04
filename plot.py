"""
Blasius laminar flat-plate similarity solutions vs SU2 and ADflow CFD
results, comparing all three ADflow grid-convergence mesh levels
(l0, l1, l2) against SU2 and Blasius theory.

Case: SU2 "Laminar Flat Plate" tutorial (lam_flatplate.cfg), run three times:
  su2/compressible_adiabatic/   -> SOLVER= NAVIER_STOKES,     adiabatic wall (no heat transfer)
  su2/compressible_isothermal/  -> SOLVER= NAVIER_STOKES,     isothermal cooled wall (148.81 K)
  su2/incompressible/           -> SOLVER= INC_NAVIER_STOKES, isothermal cooled wall (148.81 K)

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

compressible_isothermal/ is compressible_adiabatic/'s mesh/solver with the
wall BC swapped from adiabatic to the same isothermal 148.81 K setpoint as
incompressible/ and ADflow, isolating the wall-BC effect from the
compressible-vs-incompressible-algorithm effect (see su2/compressible_isothermal/lam_flatplate.cfg).

SU2 curves are PCHIP-interpolated onto a fine grid and drawn as dashed
lines rather than raw markers, since ADflow contributes three overlapping
marker series.

Plus ADflow runs on the same freestream/wall spec (adflow_run.py) --
isothermal wall like the incompressible SU2 case, compared against both
the Cf/velocity theory curves and the incompressible case's Nu(x/L).
Three mesh levels from the grid-convergence study are included, from
gc_study/, each roughly doubling the streamwise wall-cell count of the
one below it:
  l0 -> meshes/fp_l0_rebunch_fixed_inout.cgns  (156 wall cells)
  l1 -> meshes/fp_l1_rebunch_fixed_inout.cgns  (78 wall cells)
  l2 -> meshes/fp_l2_rebunch_fixed_inout.cgns  (39 wall cells)

Reads (from each SU2 case directory):
  lam_flatplate.cfg   -> freestream conditions (M, T, Re, L)
  surface_flow.vtu    -> wall skin-friction / heat-flux distribution
  flow.vtu             -> volume field, for velocity profiles

Reads (from each ADflow case directory gc_study/l0, l1, l2):
  flat_plate_heat_000_surf.cgns   -> wall SkinFriction, Pressure, Density,
                                      StantonNumber (see NU_ADFLOW below)
  flat_plate_heat_000_vol.cgns    -> volume Velocity field, for the exit
                                      velocity profile
(gc_study/l*/wall_heatflux.npz is stale/corrupted -- coords/heatFlux are
byte-identical denormalized garbage across all three levels -- so this
script reads the CGNS surface files directly instead.)

Produces:
  grid_convergence.png   Cf(x), exit-plane u/U profile, and Nu(x/L)
                          SU2 (compressible adiabatic/isothermal +
                          incompressible, dashed) + ADflow l0/l1/l2 vs theory
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.interpolate import PchipInterpolator
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
# 2b) Interpolation helper: shape-preserving (PCHIP) interpolation of a
#     scattered (x, y) SU2 series onto a fine grid for a smooth dashed
#     line, deduping any repeated x first (PCHIP needs strictly
#     increasing x). PCHIP (not a plain cubic spline) is used because it
#     never overshoots between samples -- a plain cubic spline rings
#     badly across the near-constant, slightly noisy free-stream plateau
#     in the velocity profile (u/U ~ 1 for most of the probe line).
# ----------------------------------------------------------------------
def interp_dashed(x, y, n=300, xlim=None):
    x_u, idx = np.unique(x, return_index=True)
    y_u = y[idx]
    if xlim is not None:
        keep = (x_u >= xlim[0]) & (x_u <= xlim[1])
        x_u, y_u = x_u[keep], y_u[keep]
    if len(x_u) < 3:
        return x_u, y_u
    f = PchipInterpolator(x_u, y_u)
    x_fine = np.linspace(x_u.min(), x_u.max(), n)
    return x_fine, f(x_fine)

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
    # incompressible drawn first (solid-ish, but dashed line), compressible
    # on top -- keeps both visible where curves coincide
    "incompressible": dict(color="blue", label="SU2 Incompressible (Sutherland)",
                            lw=1.8, ls="--", zorder=3),
    "compressible_adiabatic": dict(color="red", label="SU2 Compressible (Adiabatic)",
                                    lw=1.8, ls="--", zorder=4),
    # Same compressible (density-based) solver and mesh as
    # "compressible_adiabatic" above, but with the wall switched from
    # adiabatic (MARKER_HEATFLUX=0) to isothermal at 148.81 K -- the same
    # cooled-wall temperature as
    # "incompressible" and ADflow. Isolates the effect of the compressible
    # vs. ADflow algorithm with the wall thermal BC held fixed, instead of
    # comparing across both a different wall BC and a different algorithm
    # at once (see su2/compressible_isothermal/lam_flatplate.cfg).
    "compressible_isothermal": dict(color="darkviolet", label="SU2 Compressible (Isothermal)",
                                     lw=1.8, ls="--", zorder=4.5),
}

for name, case in CASES.items():
    case["wall"] = read_vtu(f"su2/{name}/surface_flow.vtu")
    case["vol"] = read_vtu(f"su2/{name}/flow.vtu")

# ----------------------------------------------------------------------
# 3b) ADflow cases -- isothermal wall at the same T_WALL_INC, read via
#     VTK's CGNS reader instead of the .vtu reader used for SU2. Three
#     mesh levels from the grid-convergence study are compared: l0
#     (finest), l1, l2 (coarsest), all archived under gc_study/.
# ----------------------------------------------------------------------
ADFLOW_CASES = {
    # l0's mfc/mec/mew below (hollow) are used for the velocity-profile
    # markers (ax2); wall_mfc/wall_mec/wall_mew (solid, same ms diameter)
    # are used for the Cf and Nusselt-number markers (ax1, ax3) instead,
    # matching the solid-dot style of l1/l2.
    "l0": dict(dir="gc_study/l0", color="green", label="ADflow (l0)",
               ms=5.0, mfc="none", mec="green", mew=0.8, alpha=0.95, zorder=7,
               wall_ms=3.5, wall_mfc="green", wall_mec="white", wall_mew=0.6),
    "l1": dict(dir="gc_study/l1", color="magenta", label="ADflow (l1)",
               ms=3.5, mfc="magenta", mec="white", mew=0.6, alpha=0.9, zorder=6),
    "l2": dict(dir="gc_study/l2", color="darkorange", label="ADflow (l2)",
               ms=3.5, mfc="darkorange", mec="white", mew=0.6, alpha=0.9, zorder=5),
}

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

def load_adflow(case_dir):
    wall = cgns_zone_by_name(read_cgns(f"{case_dir}/flat_plate_heat_000_surf.cgns"), "Isothermal")

    cc = vtk.vtkCellCenters()
    cc.SetInputData(wall)
    cc.Update()
    x = vtk_to_numpy(cc.GetOutput().GetPoints().GetData())[:, 0]
    order = np.argsort(x)
    x = x[order]
    mask = x > 1e-6

    wall_cd = wall.GetCellData()
    cf = vtk_to_numpy(wall_cd.GetArray("SkinFriction"))[order, 0]
    # StantonNumber, Pressure and Density are all in ADflow's internal
    # nondimensional units (freestream pInf = rhoInf = 1 by construction --
    # ADflow's reference pressure/density *are* the dimensional freestream
    # static P, rho), which is what the wall heat flux is recovered from
    # below (see NU_ADFLOW).
    ch = vtk_to_numpy(wall_cd.GetArray("StantonNumber"))[order]
    P_nd = vtk_to_numpy(wall_cd.GetArray("Pressure"))[order]
    rho_nd = vtk_to_numpy(wall_cd.GetArray("Density"))[order]

    vol = cgns_zone_by_name(read_cgns(f"{case_dir}/flat_plate_heat_000_vol.cgns"), "blk")
    vol_c2p = vtk.vtkCellDataToPointData()
    vol_c2p.SetInputData(vol)
    vol_c2p.Update()
    vol_pd = vol_c2p.GetOutput()

    return dict(x=x, mask=mask, cf=cf, ch=ch, P_nd=P_nd, rho_nd=rho_nd, vol_pd=vol_pd)

for name, case in ADFLOW_CASES.items():
    case.update(load_adflow(case["dir"]))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

# ----------------------------------------------------------------------
# 4) Skin friction: SU2 (both cases, interpolated dashed) vs Blasius
#    Cf(x) = 0.664/sqrt(Re_x)
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

    x_fine, cf_fine = interp_dashed(x_wall_m, cf_su2)
    ax1.plot(
        x_fine, cf_fine, ls=case["ls"],
        color=case["color"], lw=case["lw"],
        alpha=0.9, zorder=case["zorder"], label=case["label"],
    )

for name, case in ADFLOW_CASES.items():
    ax1.plot(
        case["x"][case["mask"]], case["cf"][case["mask"]], "o",
        ms=case.get("wall_ms", case["ms"]), mfc=case.get("wall_mfc", case["mfc"]),
        mec=case.get("wall_mec", case["mec"]), mew=case.get("wall_mew", case["mew"]),
        alpha=case["alpha"], zorder=case["zorder"], label=case["label"],
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
# 5) Velocity profiles u/U_inf vs eta at the exit plane (x = L), both
#    SU2 cases interpolated dashed, ADflow levels as markers
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
    # compressible (both adiabatic and isothermal) Velocity field is
    # dimensional (m/s); incompressible Velocity field is already
    # non-dimensionalized by U_inf
    u_over_U = u_raw / U_INF if name != "incompressible" else u_raw
    eta_probe = y * np.sqrt(U_INF / (NU_INF * x_val))
    eta_fine, u_fine = interp_dashed(eta_probe, u_over_U, n=300, xlim=(0, 9))
    ax2.plot(
        u_fine, eta_fine, ls=case["ls"],
        color=case["color"], lw=case["lw"],
        alpha=0.9, zorder=case["zorder"], label=case["label"],
    )

# ADflow: Velocity is nondimensionalized by sqrt(pRef/rhoRef), not U_inf,
# so u/U_inf = u_nondim / (Mach * sqrt(gamma)) rather than a raw copy.
for name, case in ADFLOW_CASES.items():
    y_adflow, u_adflow_nd = probe_column(case["vol_pd"], x_val - 1e-4, y_max)
    u_over_U_adflow = u_adflow_nd / (MACH * np.sqrt(GAMMA))
    eta_adflow = y_adflow * np.sqrt(U_INF / (NU_INF * x_val))
    idx_adflow = thin_by_eta(eta_adflow)
    ax2.plot(
        u_over_U_adflow[idx_adflow], eta_adflow[idx_adflow], "o",
        ms=case["ms"], mfc=case["mfc"], mec=case["mec"], mew=case["mew"],
        alpha=case["alpha"], zorder=case["zorder"],
        label=case["label"],
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
# 6) Nusselt number vs x/L. Only the two isothermal SU2 cases have wall
#    heat transfer to compare -- "compressible_adiabatic" (adiabatic,
#    MARKER_HEATFLUX=0) is skipped here. Nu_x = q_wall*x/(k*(T_inf-T_wall)),
#    with k = K_INC (freestream-viscosity/constant-Prandtl conductivity)
#    and dT held at the constant MARKER_ISOTHERMAL setpoint for all
#    curves, for a consistent Nu_x definition even though the compressible
#    solves' local k and wall T both vary slightly along x (Sutherland).
# ----------------------------------------------------------------------
dT = T_INF - T_WALL_INC  # K

def su2_wall_Nu(case):
    wall_pts = vtk_to_numpy(case["wall"].GetPoints().GetData())
    q_wall = vtk_to_numpy(case["wall"].GetPointData().GetArray("Heat_Flux"))

    order = np.argsort(wall_pts[:, 0])
    x_wall = wall_pts[order, 0]
    q_wall = q_wall[order]

    mask = x_wall > 1e-6
    x_wall_m = x_wall[mask]
    q_wall_m = q_wall[mask]

    Nu = q_wall_m * x_wall_m / (K_INC * dT)
    xoL = x_wall_m / L_REF
    return xoL, Nu

xoL_theory = np.linspace(1e-4, 1.0, 200)
Re_x_theory = U_INF * (xoL_theory * L_REF) / NU_INF
Nu_theory = 0.332 * np.sqrt(Re_x_theory) * PRANDTL_LAM ** (1.0 / 3.0)

ax3.plot(
    xoL_theory, Nu_theory, "-",
    color="black", lw=1.5, zorder=2,
    label=r"Blasius: $0.332\sqrt{Re_x}\,Pr^{1/3}$",
)
for name in ("incompressible", "compressible_isothermal"):
    case = CASES[name]
    xoL_su2, Nu_su2 = su2_wall_Nu(case)
    xoL_fine, Nu_fine = interp_dashed(xoL_su2, Nu_su2)
    ax3.plot(
        xoL_fine, Nu_fine, ls=case["ls"],
        color=case["color"], lw=case["lw"],
        alpha=0.9, zorder=case["zorder"], label=case["label"],
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
P_INF = RHO_INF * R_GAS * T_INF  # ideal gas static freestream pressure == ADflow's pRef
scaleDim = P_INF * np.sqrt(P_INF / RHO_INF)

for name, case in ADFLOW_CASES.items():
    a2_nd = GAMMA * case["P_nd"] / case["rho_nd"]
    qw_nd = case["ch"] * fact_nd * (a2Tot_nd - a2_nd)
    qw_adflow = qw_nd * scaleDim  # W/m^2

    Nu_adflow = qw_adflow[case["mask"]] * case["x"][case["mask"]] / (K_INC * dT)
    xoL_adflow = case["x"][case["mask"]] / L_REF

    ax3.plot(
        xoL_adflow, Nu_adflow, "o",
        ms=case.get("wall_ms", case["ms"]), mfc=case.get("wall_mfc", case["mfc"]),
        mec=case.get("wall_mec", case["mec"]), mew=case.get("wall_mew", case["mew"]),
        alpha=case["alpha"], zorder=case["zorder"], label=case["label"],
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

fig.suptitle("SU2 (lam_flatplate) + ADflow grid-convergence (l0/l1/l2) vs Blasius similarity solutions")
fig.tight_layout()
fig.savefig("grid_convergence.png", dpi=300)
print("Wrote grid_convergence.png")
