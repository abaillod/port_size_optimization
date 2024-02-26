import os
from pathlib import Path
import numpy as np
import time
import simsopt
from scipy.optimize import minimize
from simsopt._core.optimizable import load
from simsopt.geo import ArclengthVariation
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.objectives.utilities import QuadraticPenalty
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves, create_equally_spaced_windowpane_curves
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, coils_via_symmetries, ScaledCurrent, apply_symmetries_to_curves
from simsopt.geo.curveobjectives import CurveLength, CurveCurveDistance, \
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber

import matplotlib.pyplot as plt
import git
from simsopt.geo import CurveCWSFourier
from simsopt.geo import EnclosedXYArea, CurveCurveXYdistance, CurveXYConvexity, ToroidalAngleConstraint, UpwardFacingPort

import argparse
import importlib
import datetime

# Read command line arguments
parser = argparse.ArgumentParser()

# If ran with "--pickle", expect the input to be a pickle.
parser.add_argument("--pickle", dest="pickle", default=False, action="store_true")

# Provide input as a relative or absolute path
parser.add_argument("--input", dest="input", default=None)

# Prepare args
args = parser.parse_args()

# Read input dict
if args.pickle:
    with open(args.input, 'rb') as f:
        inputs = pickle.load(f)
else:
    fname = args.input.replace('/','.')
    if fname[-3:]=='.py':
        fname = fname[:-3]
    std = importlib.import_module(fname, package=None)
    inputs = std.inputs


date = datetime.datetime

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = inputs['ncoils']

# Major radius for the initial circular coils:
R0 = inputs['R0']

# Minor radius for the initial circular coils:
R1 = inputs['R1']

# Number of Fourier modes describing each Cartesian component of each coil:
order = inputs['order']

# Weight on the curve lengths in the objective function. We use the `Weight`
# class here to later easily adjust the scalar value and rerun the optimization
# without having to rebuild the objective.
LENGTH_WEIGHT = inputs['LENGTH_WEIGHT']

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = inputs['CC_THRESHOLD']
CC_WEIGHT = inputs['CC_WEIGHT']

# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = inputs['CS_THRESHOLD']
CS_WEIGHT = inputs['CS_WEIGHT']

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = inputs['CURVATURE_THRESHOLD']
CURVATURE_WEIGHT = inputs['CURVATURE_WEIGHT']

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = inputs['MSC_THRESHOLD']
MSC_WEIGHT = inputs['MSC_WEIGHT']

# Port size relevant weights
wdd = inputs['PORT_COIL_DISTANCE_WEIGHT']
#wco = 0
#wph = 1E3
warc = inputs['PORT_ARCLENGTH_WEIGHT']
wufp = inputs['PORT_UPWARD_WEIGHT']
wport = inputs['PORT_WEIGHT']

# Number of iterations to perform:
MAXITER =  inputs['MAXITER']

# File for the desired boundary magnetic surface:
filename = inputs['filename']

# Directory for output
OUT_DIR = f"./output/" + inputs['output_directory']
os.makedirs(OUT_DIR, exist_ok=False)

# Resolution for surface in real space
nphi = inputs['nphi']
ntheta = inputs['ntheta']

# Read port related input
port_order = inputs['port_order']
qpts = inputs['port_quadpoints']
iphi0 = inputs['port_iphi0']

# create log

repo = git.Repo(simsopt.__path__[0], search_parent_directories=True)
sha0 = repo.head.object.hexsha

repo = git.Repo(search_parent_directories=True)
sha1 = repo.head.object.hexsha

with open(os.path.join(OUT_DIR, 'log.txt'), 'w') as f:
    f.write('VERTICAL PORT SIZE OPTIMIZATION\n')
    f.write(f"Using simsopt version {sha0}\n")
    f.write(f"Using port size optimization git version {sha1}\n")
    f.write(f"Date = {date.date(date.now()).isoformat()} at {date.now().strftime('%Hh%M')}\n")
    
def logprint(s):
    with open(os.path.join(OUT_DIR, 'log.txt'), 'a') as f:
        f.write(s)

# ----------------------------------------------------------------------------------------------------
#                                       GENERATE COILS
# ====================================================================================================
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
fulls = SurfaceRZFourier.from_vmec_input(filename, nphi=nphi, ntheta=ntheta)

# Create the initial coils:
base_tf_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=512) 
base_tf_currents = [Current(-1e4) for i in range(ncoils)]

base_curves = base_tf_curves
base_currents = base_tf_currents

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)

for c in coils:
    c.unfix_all()

bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))

curves = [c.curve for c in coils]
curves_to_vtk(curves, os.path.join(OUT_DIR, "curves_initial"))
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(os.path.join(OUT_DIR, "surf_initial"), extra_data=pointData)



# ----------------------------------------------------------------------------------------------------
#                                       GENERATE PORT
# ====================================================================================================
order = port_order
curve_cws = CurveCWSFourier(
    qpts,
    order,
    s
)
nfp = s.nfp
dphi = 1/(2*nfp) * 1/ncoils
curve_cws.set('phic(0)', iphi0*dphi)
curve_cws.set('phic(1)', dphi/3.0)
curve_cws.set('thetac(0)', 0.25)
curve_cws.set('thetas(1)', 0.1)

curves_to_vtk([curve_cws], os.path.join(OUT_DIR, "port_initial"))
curve_cws.save(os.path.join(OUT_DIR, "port_initial.json"))

# ----------------------------------------------------------------------------------------------------
#                                    DEFINE OBJECTIVE FUNCTION
# ====================================================================================================
# Define the individual terms objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
linkNum = LinkingNumber(base_tf_curves)

# Port relavant penalties
Jxyarea = EnclosedXYArea( curve_cws )
Jccxydist = CurveCurveXYdistance( curves, curve_cws, 0.05 )
Jconvex = CurveXYConvexity( curve_cws )
Jarc = ArclengthVariation( curve_cws )
Jufp = UpwardFacingPort(curve_cws)
Jport = -1 * Jxyarea + wdd * Jccxydist + warc * Jarc + wufp * Jufp #+ wco * Jconvex


# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
JF = Jf \
    + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls), 2.6*ncoils) \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs) \
    + QuadraticPenalty(linkNum, 0.1) \
    + wport * Jport

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    MaxBdotN = np.max(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    mean_AbsB = np.mean(bs.AbsB())

    
    A = Jxyarea.J()
    CC = Jccxydist.J()
    arc = Jarc.J()
    ufp = Jufp.J()
    #convex = Jconvex.J()
    
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}, Jport={Jport.J():.2E}\n"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    outstr += f", ⟨B·n⟩/|B|={BdotN/mean_AbsB:.1e}"
    outstr += f", (Max B·n)/|B|={MaxBdotN/mean_AbsB:.1e}"
    outstr += f", Link Number = {linkNum.J()}\n"
    outstr += f"Area={A:.2E}, Coil-coil dist={CC:.2E}, Arc penalty={arc:.2E}, UFP={ufp:.2E}\n\n"
    logprint(outstr)
    return J, grad


# ----------------------------------------------------------------------------------------------------
#                                    RUN OPTIMIZATION
# ====================================================================================================
dofs = JF.x
J, _ = fun(dofs)
logprint('\n\n')
logprint(f'INITIAL FUNCTION VALUE IS {J:.5E}')

res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-12)
logprint(res.message)

dofs = JF.x
J, _ = fun(dofs)
logprint('\n\n')
logprint(f'FINAL FUNCTION VALUE IS {J:.5E}')


# ----------------------------------------------------------------------------------------------------
#                                    PREPARING OUTPUT
# ====================================================================================================
curves_to_vtk(curves,  os.path.join(OUT_DIR, f"curves_final"))
curves_to_vtk([curve_cws], os.path.join(OUT_DIR, "port_final"))
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(os.path.join(OUT_DIR, "surf_final"), extra_data=pointData)
curve_cws.save(os.path.join(OUT_DIR, "port_final.json"))
bs.save(os.path.join(OUT_DIR, f'biotsavart_final.json'))