import os
from pathlib import Path
import numpy as np
import time
import simsopt
from scipy.optimize import minimize
from simsopt.util import comm_world
from simsopt._core.optimizable import load
from simsopt.geo import ArclengthVariation
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.objectives.utilities import QuadraticPenalty
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves, create_equally_spaced_windowpane_curves
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, coils_via_symmetries, ScaledCurrent, apply_symmetries_to_curves, apply_symmetries_to_currents, Coil
from simsopt.geo.curveobjectives import CurveLength, CurveCurveDistance, \
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber

import matplotlib.pyplot as plt
import git
from simsopt.geo import CurveCWSFourier, Curve2D
from simsopt.geo import ProjectedEnclosedArea, ProjectedCurveCurveDistance, ProjectedCurveConvexity, DirectedFacingPort

import argparse
import importlib
import datetime
import pickle
from simsopt.objectives import Weight


date = datetime.datetime

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 5

# Major radius for the initial circular coils:
R0 = 1.0

# Minor radius for the initial circular coils:
R1 = 0.5

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5

# Weight on the curve lengths in the objective function. We use the `Weight`
# class here to later easily adjust the scalar value and rerun the optimization
# without having to rebuild the objective.
LENGTH_WEIGHT = Weight(0.0156434465)

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.08
CC_WEIGHT = Weight(156.434465)

# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0.12
CS_WEIGHT = Weight(1564.34465)

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 12.
CURVATURE_WEIGHT = Weight(0.000000156434465)

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 11
MSC_WEIGHT = Weight(1.10e-08)

LNUM_WEIGHT = Weight(0.1)

# ---------------------------- PORT STUFF -----------------------------------------------
# Port size relevant weights
port_order = 2
port_qpts = 128

wport = Weight(1)
wdd = Weight(1E0)
warc = Weight(1E-2)
wufp = Weight(1E-2)

# Read port related input
port_order = 2
qpts = 128
iphi0 = 0

# --------------------------------- WPs STUFF ---------------------------------------------
nwp_theta = 12
nwp_phi = 6
wp_order = 1
wp_qpts = 32


# Number of iterations to perform:
MAXITER_I =  1E2
MAXITER_II = 1E4

# File for the desired boundary magnetic surface:
filename = '/burg/apam/users/ab5667/Projects/port_size_optimization/configurations/qh_landreman_paul/input.LandremanPaul2021_QH'

# Directory for output
OUT_DIR = f"./output"
os.makedirs(OUT_DIR, exist_ok=True)

# Resolution for surface in real space
nphi = 200
ntheta = 32


# create log
repo = git.Repo("~/Github/simsopt")
sha0 = repo.head.object.hexsha

repo = git.Repo(search_parent_directories=True)
sha1 = repo.head.object.hexsha

with open('log.txt', 'w') as f:
    f.write('HORIZONTAL PORT SIZE OPTIMIZATION WITH WINDOWPANE\n')
    f.write(f"Using simsopt version {sha0}\n")
    f.write(f"Using port size optimization git version {sha1}\n")
    f.write(f"Date = {date.date(date.now()).isoformat()} at {date.now().strftime('%Hh%M')}\n")
    
def logprint(s):
    if comm_world.rank == 0:
        with open('log.txt', 'a') as f:
            f.write(s + "\n")
        print(s)


# ----------------------------------------------------------------------------------------------------
#                                       GENERATE TF COILS
# ====================================================================================================
logprint("Generating TF coils...")

s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
fulls = SurfaceRZFourier.from_vmec_input(filename, nphi=nphi, ntheta=ntheta)
v = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
v.extend_via_normal(0.1)

s.fix_all()
fulls.fix_all()

# Create the initial coils:
base_tf_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=512) 
base_tf_currents = [Current(-1e4) for i in range(ncoils)]

tf_coils = coils_via_symmetries(base_tf_curves, base_tf_currents, s.nfp, True)

for c in tf_coils:
    c.unfix_all()
base_tf_currents[0].fix_all()

# Save curves
tf_curves = [c.curve for c in tf_coils]
curves_to_vtk(tf_curves, os.path.join(OUT_DIR, "tf_curves_initial"))


# ----------------------------------------------------------------------------------------------------
#                                       GENERATE WPs COILS
# ====================================================================================================
logprint("Generating WP coils...")
c2d = []
base_wp_curve = []
tt = np.linspace(0,1,nwp_theta,endpoint=False)
pp = np.linspace(0,1/(2*s.nfp),nwp_phi,endpoint=False)
dt = tt[1]-tt[2]
dp = pp[1]-pp[2]
for it in tt:
    for ip in pp:
        c2d.append(
            Curve2D(wp_qpts, wp_order)
        )
        base_wp_curve.append(
            CurveCWSFourier(c2d[-1], v)
        )
        
        base_wp_curve[-1].curve2d.set('phic(0)', ip-dp/2.0)
        base_wp_curve[-1].curve2d.set('phic(1)', dp/3)
        base_wp_curve[-1].curve2d.set('thetac(0)', it)
        base_wp_curve[-1].curve2d.set('thetas(1)', dt/3)

wp_curves = apply_symmetries_to_curves( base_wp_curve, v.nfp, True)
for c in wp_curves:
    c.fix_all()

base_wp_current = [ScaledCurrent(Current(0), 1e5) for c in base_wp_curve]
for c in base_wp_current:
    c.unfix_all()
wp_currents = apply_symmetries_to_currents( base_wp_current, v.nfp, True)

wp_coils = [Coil(curve, current) for curve, current in zip(wp_curves, wp_currents)]

# Save curves
curves_to_vtk(wp_curves, os.path.join(OUT_DIR, "wp_curves_initial"))

# ----------------------------------------------------------------------------------------------------
#                                       BIOTSAVART OBJECT
# ====================================================================================================
logprint("Generating BiotSavart object...")
full_coils = tf_coils + wp_coils
bs = BiotSavart(full_coils)
bs.set_points(s.gamma().reshape((-1, 3)))

pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(os.path.join(OUT_DIR, "surf_initial"), extra_data=pointData)
bs.save(os.path.join(OUT_DIR, "biotsavart_initial.json"))


# ----------------------------------------------------------------------------------------------------
#                                       GENERATE PORT
# ====================================================================================================
logprint("Generating access port...")
c2dport = Curve2D(port_qpts, port_order )
port_curve = CurveCWSFourier(
    c2dport,
    s
)
nfp = s.nfp
dphi = 1/(2*nfp) * 1/ncoils
port_curve.curve2d.set('phic(0)', iphi0*dphi)
port_curve.curve2d.set('phic(1)', dphi/3.0)
port_curve.curve2d.set('thetac(0)', 0.0)
port_curve.curve2d.set('thetas(1)', 0.05)

full_port_curves = apply_symmetries_to_curves( [port_curve], s.nfp, True )

curves_to_vtk(full_port_curves, os.path.join(OUT_DIR, "port_initial"))
port_curve.save(os.path.join(OUT_DIR, "port_initial.json"))

# ----------------------------------------------------------------------------------------------------
#                                    DEFINE OBJECTIVE FUNCTION
# ====================================================================================================
logprint("Defining objective function...")
# Define the individual terms objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_tf_curves]
Jccdist = CurveCurveDistance(tf_curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(base_tf_curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_tf_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_tf_curves]
linkNum = LinkingNumber(base_tf_curves)

# Port relavant penalties
Jxyarea = ProjectedEnclosedArea( port_curve, projection='zphi' )
Jccxydist = ProjectedCurveCurveDistance( tf_curves, port_curve, 0.05, projection='zphi' )
Jconvex = ProjectedCurveConvexity( port_curve, projection='zphi' )
Jarc = ArclengthVariation( port_curve )
Jufp = DirectedFacingPort(port_curve, projection='r')

Jport = -1 * Jxyarea + wdd * Jccxydist + warc * Jarc + wufp * Jufp #+ wco * Jconvex


# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
Jlength = QuadraticPenalty(sum(Jls), 2.6*ncoils)
J_sumcs = sum(Jcs)
J_summscs = sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs)
Jln = QuadraticPenalty(linkNum, 0.1)
JF = Jf \
    + LENGTH_WEIGHT * Jlength \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * J_sumcs \
    + MSC_WEIGHT * J_summscs \
    + LNUM_WEIGHT * Jln \
    + wport * Jport

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize


def fun(dofs, info={'Nfeval': 0, 'print': False}):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    MaxBdotN = np.max(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    mean_AbsB = np.mean(bs.AbsB())

    
    if info['print']:
        jf = Jf.J()
        BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        MaxBdotN = np.max(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        mean_AbsB = np.mean(bs.AbsB())
    
        
        A = Jxyarea.J()
        CC = Jccxydist.J()
        arc = Jarc.J()
        
        outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in full_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        outstr += f", ⟨B·n⟩/|B|={BdotN/mean_AbsB:.1e}"
        outstr += f", (Max B·n)/|B|={MaxBdotN/mean_AbsB:.1e}"
        #outstr += f", Link Number = {linkNum.J()}\n"
        outstr += f"Jport={Jport.J():.2E}, Area={A:.2E}, Coil-coil dist={CC:.2E}, Arc penalty={arc:.2E}"#, Convex={convex:.2E}"
        logprint(outstr)
    return J, grad


# ----------------------------------------------------------------------------------------------------
#                                    FIND WEIGHTS
# ====================================================================================================
logprint("\n")
logprint("================================================================")
dofs0 = JF.x

logprint("Evaluating initial objective...")
J, dJ = fun(dofs0)
logprint(f"J = {J:.4E}")

logprint("Starting iterations to find penalties weight...")
satisfied = False
counter=1
while not satisfied:
    logprint(f'Weight iteration {counter}:: Running with wdd={wdd.value:.2E}, warc={warc.value:.2E}, wufp={wufp.value:.2E}, Length weight={LENGTH_WEIGHT.value:.2E}, CC_WEIGHT={CC_WEIGHT.value:.2E}, CS_WEIGHT={CS_WEIGHT.value:.2E}, CURVATURE_WEIGHT={CURVATURE_WEIGHT.value:.2E}, MSC_WEIGHT={MSC_WEIGHT.value:.2E}, LNUM_WEIGHT={LNUM_WEIGHT.value:.2E}')
    MAXITER = MAXITER_I
    dofs = dofs0
    res = minimize(fun, dofs, jac=True, args=({'Nfeval': 0, 'print':False}), method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-12)

    satisfied = True

    jl = Jlength.J()
    ccdist = Jccdist.J()
    csdist = Jcsdist.J()
    cpdist = Jccxydist.J()
    sumcs = J_sumcs.J() 
    summscs = J_summscs.J()
    jln = Jln.J()
    arcpen = Jarc.J()
    ufppen = Jufp.J()
    logprint(f'length penalty={jl:.2E}, ccdist={ccdist:.2E}, csdist={csdist:.2E}, sumcs={sumcs:.2E}, summscs={summscs:.2E}, jln={jln:.2E}, cpdist={cpdist:.2E}, arcpen={arcpen:.2E}, ufp={ufppen:.2E}')

    if jl > 1:
        logprint('coils too long')
        LENGTH_WEIGHT *= 1.5
        satisfied = False

    if ccdist > 1E-4:
        logprint('coils too close to one another')
        CC_WEIGHT *= 1.5
        satisfied = False

    if csdist > 1E-4:
        logprint('coils too close to surface')
        CS_WEIGHT *= 1.5
        satisfied = False

    if sumcs > 1:
        logprint('coils max curvature exceeded')
        CURVATURE_WEIGHT *= 1.5
        satisfied = False

    if summscs > 1:
        logprint('coils mean curvature exceeded')
        MSC_WEIGHT *= 1.5
        satisfied = False
        
    if jln > 1E-3:
        logprint('Coils are probably linked')
        LNUM_WEIGHT *= 1.5
        satisfied = False
        
    if cpdist>1E-6:
        logprint('cpdist too large')
        wdd *= 1.5
        satisfied = False

    if arcpen>1:
        logprint('arcpen too large')
        warc *= 1.5
        satisfied = False

    if ufppen>= 0.01:
        logprint('ufppen too large')
        wufp *= 1.5
        satisfied = False

    counter += 1

    logprint("")

# ----------------------------------------------------------------------------------------------------
#                                    RUN OPTIMIZATION
# ====================================================================================================
logprint("================================================================")
logprint("Starting optimization")
dofs = JF.x
J, _ = fun(dofs)
logprint('')
logprint(f'INITIAL FUNCTION VALUE IS {J:.5E}')

res = minimize(fun, dofs, jac=True, args=({'Nfeval': 0, 'print':True}), method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-12)
logprint(res.message)

dofs = JF.x
J, _ = fun(dofs)
logprint('\n\n')
logprint(f'FINAL FUNCTION VALUE IS {J:.5E}')


# ----------------------------------------------------------------------------------------------------
#                                    PREPARING OUTPUT
# ====================================================================================================
curves_to_vtk(curves,  os.path.join(OUT_DIR, f"curves_final"))
curves_to_vtk(wp_curves, os.path.join(OUT_DIR, "wp_curves_final"))
curves_to_vtk(full_port_curves, os.path.join(OUT_DIR, "port_final"))
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(os.path.join(OUT_DIR, "surf_final"), extra_data=pointData)
port_curve.save(os.path.join(OUT_DIR, "port_final.json"))
bs.save(os.path.join(OUT_DIR, f'biotsavart_final.json'))
