import os
from pathlib import Path
import numpy as np
import time
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

from simsopt.geo import SurfaceSurfaceDistance
from simsopt.field import SurfaceClassifier, \
    particles_to_vtk, compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data
from simsopt.objectives import Weight


import matplotlib.pyplot as plt

from simsopt.geo import CurveCWSFourier
from simsopt.geo import Curve2D
from simsopt.geo import ProjectedEnclosedArea, ProjectedCurveCurveDistance, ProjectedCurveConvexity, DirectedFacingPort

from pystellplot.Paraview import coils_to_vtk, surf_to_vtk

import git
import datetime
import simsopt

date = datetime.datetime

os.makedirs('output', exist_ok=True)

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

# Port size relevant weights
port_order = 2
port_qpts = 128

wport = Weight(1)
wdd = Weight(1E0)
warc = Weight(1E-2)
wufp = Weight(1E-2)

# Number of iterations to perform:
MAXITER_I =  1E2
MAXITER_II = 1E4

# File for the desired boundary magnetic surface:
filename = '/Users/antoinebaillod/Projects/Accessibilty/configurations/qh_landreman_paul/wiedman_coils/input.LandremanPaul2021_QH'
#filename = 'input.nfp2_QA'

# Directory for output
OUT_DIR = f"./output/NoWPs_{date.date(date.now()).isoformat()}_{date.now().strftime('%Hh%M')}"
os.makedirs(OUT_DIR, exist_ok=True)


# create log
repo = git.Repo(simsopt.__path__[0], search_parent_directories=True)
sha0 = repo.head.object.hexsha

repo = git.Repo(search_parent_directories=True)
sha1 = repo.head.object.hexsha

date = datetime.datetime
with open(os.path.join(OUT_DIR, 'log.txt'), 'w') as f:
    f.write('HORIZONTAL PORT SIZE OPTIMIZATION\n')
    f.write(f"Using simsopt version {sha0}\n")
    f.write(f"Using port size optimization git version {sha1}\n")
    f.write(f"Date = {date.date(date.now()).isoformat()} at {date.now().strftime('%Hh%M')}\n")
    
def logprint(s):
    with open(os.path.join(OUT_DIR, 'log.txt'), 'a') as f:
        f.write(s)
    print(s)

# INITIALIZATION
# =======================
# Initialize the boundary magnetic surface:
nphi = 200
ntheta = 32
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
fulls = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=nphi, ntheta=ntheta)
s.fix_all()
fulls.fix_all()

# Create the initial tf coils:
base_tf_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=512) 
base_tf_currents = [Current(1e5) for _ in range(ncoils)]

tf_coils = coils_via_symmetries(base_tf_curves, base_tf_currents, s.nfp, True)
tf_curves = [c.curve for c in tf_coils]

for c in tf_coils:
    c.unfix_all()
tf_coils[0].current.fix_all()

#Define one port
c2dport = Curve2D(port_qpts, port_order )
port_curve = CurveCWSFourier(
    c2dport,
    s
)

c2dport.fix_all()

full_port_curves = apply_symmetries_to_curves( [port_curve], s.nfp, True )

nfp = s.nfp
dphi = 1/(2*nfp) * 1/ncoils
iphi0 = 0
port_curve.curve2d.set('phic(0)', iphi0*dphi)
port_curve.curve2d.set('phic(1)', dphi/3.0)
port_curve.curve2d.set('thetac(0)', 0.0)
port_curve.curve2d.set('thetas(1)', 0.05)


full_coils = tf_coils
full_curves = [c.curve for c in full_coils]

bs = BiotSavart(full_coils)
bs.set_points(s.gamma().reshape((-1, 3)))

curves_to_vtk(tf_curves, os.path.join(OUT_DIR, "tf_curves_init"))
curves_to_vtk(full_port_curves, os.path.joinr(OUT_DIR,"port_curves_init"))
surf_to_vtk(os.path.join(OUT_DIR, "plasma_bnd_init"), bs, s)


# OBJECTIVE FUNCTION
#==========================================
# Define the individual terms objective function:
## Quad flux and coil reguularization
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_tf_curves]
Jccdist = CurveCurveDistance(tf_curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(base_tf_curves, s, CS_THRESHOLD)
#Jcwsdist = CurveSurfaceDistance(base_tf_curves, v, 0.05)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_tf_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_tf_curves]
linkNum = LinkingNumber(tf_curves)

Jlength = QuadraticPenalty(sum(Jls), 2.6*ncoils, 'max')
Jsum_cs = sum(Jcs)
Jsum_mscs = sum(QuadraticPenalty(J, MSC_THRESHOLD, 'max') for J in Jmscs)
Jlnum = QuadraticPenalty(linkNum, 0.1, 'max')

JF = Jf \
    + LENGTH_WEIGHT * Jlength \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * Jsum_cs \
    + MSC_WEIGHT * Jsum_mscs \
    + LNUM_WEIGHT * Jlnum

# Jsum_mscs Jcsdist Jccdist Jsum_cs


# Port relavant penalties
Jxyarea = ProjectedEnclosedArea( port_curve, projection='zphi' )
Jccxydist = ProjectedCurveCurveDistance( tf_curves, port_curve, 0.05, projection='zphi' )
#Jconvex = ProjectedCurveConvexity( port_curve, projection='zphi' )
Jarc = ArclengthVariation( port_curve )
Jufp = DirectedFacingPort(port_curve, projection='r')
Jport = -1 * Jxyarea + wdd * Jccxydist + warc * Jarc + wufp * Jufp #+ wco * Jconvex

JF += wport * Jport

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:

#J= Jport

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize

ports =  apply_symmetries_to_curves( [port_curve], s.nfp, False )

bs.save(os.path.join(OUT_DIR,'biotsavart_initial.json'))
port_curve.save(os.path.join(OUT_DIR,'port_initial.json'))
curves_to_vtk(ports, os.path.join(OUT_DIR,'port_initial'))
surf_to_vtk( os.path.join(OUT_DIR,"surface_initial"), bs, fulls )



def fun(dofs, info={'Nfeval': 0, 'print': False}):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()

    if info['print']:
        jf = Jf.J()
        BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        MaxBdotN = np.max(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        mean_AbsB = np.mean(bs.AbsB())
    
        
        A = Jxyarea.J()
        CC = Jccxydist.J()
        arc = Jarc.J()
        #convex = Jconvex.J()
        
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
        outstr += f"Jport={Jport.J():.2E}, Area={A:.2E}, Coil-coil dist={CC:.2E}, Arc penalty={arc:.2E}\n"#, Convex={convex:.2E}"
        logprint(outstr)
    return J, grad



# FIND WEIGHTS
# ===============================

#LENGTH_WEIGHT = Weight(0.0156434465)
#CC_WEIGHT = Weight(156.434465)
#CS_WEIGHT = Weight(1564.34465)
#CURVATURE_WEIGHT = Weight(0.000000156434465)
#MSC_WEIGHT = Weight(1.10e-08)
#LNUM_WEIGHT = Weight(0.1)

# wdd = Weight(1E0)
# warc = Weight(1E-2)
# wufp = Weight(1E-2)

dofs0 = JF.x
satisfied = False
counter=1
while not satisfied:
    #curve_cws.x = dofs0
    logprint(f'Weight iteration {counter}:: Running with wdd={wdd.value:.2E}, warc={warc.value:.2E}, wufp={wufp.value:.2E}')
    MAXITER = MAXITER_I
    dofs = dofs0
    res = minimize(fun, dofs, jac=True, args=({'Nfeval': 0, 'print':False}), method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-12)

    satisfied = True

    jl = Jlength.J()
    ccdist = Jccdist.J()
    csdist = Jcsdist.J()
    cpdist = Jccxydist.J()
    sumcs = Jsum_cs.J()
    summscs = Jsum_mscs.J()
    jln = Jlnum.J()
    arcpen = Jarc.J()
    ufppen = Jufp.J()
    logprint(f'length penalty={jl:.2E}, ccdist={ccdist:.2E}, csdist={csdist:.2E}, sumcs={sumcs:.2E}, summscs={summscs:.2E}, jln={jln:.2E}, cpdist={cpdist:.2E}, arcpen={arcpen:.2E}, ufp={ufppen:.2E}\n')

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

dofs = JF.x
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER_II, 'maxcor': 300}, tol=1e-12)

logprint(res.message)

bs.save(os.path.join(OUT_DIR,'biotsavart_final.json'))
port_curve.save(os.path.join(OUT_DIR,'port_final.json'))
curves_to_vtk(ports, os.path.join(OUT_DIR,'port_final'))
surf_to_vtk( os.path.join(OUT_DIR,"surface_final", bs, fulls ))
