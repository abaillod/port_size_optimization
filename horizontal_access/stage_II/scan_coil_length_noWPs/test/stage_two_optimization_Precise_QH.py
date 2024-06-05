#!/usr/bin/env python
r"""
In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface.  In this particular case we consider a vacuum field, so the target is
just zero.

The objective is given by

    J = (1/2) \int |B dot n|^2 ds
        + LENGTH_WEIGHT * (sum CurveLength)
        + DISTANCE_WEIGHT * MininumDistancePenalty(DISTANCE_THRESHOLD)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)

if any of the weights are increased, or the thresholds are tightened, the coils
are more regular and better separated, but the target normal field may not be
achieved as well. This example demonstrates the adjustment of weights and
penalties via the use of the `Weight` class.

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

import os
from pathlib import Path
import numpy as np
import time
from scipy.optimize import minimize
from simsopt.util import comm_world
from simsopt._core.optimizable import load
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.objectives.utilities import QuadraticPenalty
from simsopt.field.coil import apply_symmetries_to_curves
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, coils_via_symmetries, ScaledCurrent
from simsopt.geo.curveobjectives import CurveLength, CurveCurveDistance, \
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, LinkingNumber

from pystellplot.Paraview import coils_to_vtk, surf_to_vtk

from simsopt.geo import ArclengthVariation
from simsopt.objectives import Weight
from simsopt.geo import ProjectedEnclosedArea, ProjectedCurveCurveDistance, DirectedFacingPort

from simsopt.geo import CurveCWSFourier, Curve2D
from simsopt.geo import ProjectedEnclosedArea, ProjectedCurveCurveDistance, ProjectedCurveConvexity, DirectedFacingPort

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 5

# Major radius for the initial circular coils:
R0 = 1.0

# Minor radius for the initial circular coils:
R1 = 0.5

# Number of Fourier modes describing each Cartesian component of each coil:
order = 7

# Qhadratic flux weight
QFWeight = Weight( 1E5 )

# Weight on the curve lengths in the objective function. We use the `Weight`
# class here to later easily adjust the scalar value and rerun the optimization
# without having to rebuild the objective.
LENGTH_WEIGHT = Weight( 0.0156434465* 100 ) 
coil_length = 3.0

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.08
CC_WEIGHT = Weight( 156.434465 )

# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0.14 # Increased to 0.14 - this should enforce coils to be around the WPs
CS_WEIGHT = Weight( 1564.34465 )

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 12.
CURVATURE_WEIGHT = Weight( 0.000000156434465 )

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 11
MSC_WEIGHT = Weight( 1.10e-08 )

# Linking number weight
LNUM_WEIGHT = Weight( 1 )

# Arclength penalty weighty
ARCLENGTH_WEIGHT = Weight(1)

# Number of iterations to perform:
MAXITER = 1e3

# File for the desired boundary magnetic surface:
TEST_DIR = filename = '/Users/antoinebaillod/Projects/Accessibilty/configurations/qh_landreman_paul'
filename = TEST_DIR + '/input.LandremanPaul2021_QH'

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

with open('log.txt', 'w') as f:
    f.write('HORIZONTAL PORT SIZE OPTIMIZATION WITH WINDOWPANE\n')
    
def logprint(s):
    if comm_world.rank == 0:
        with open('log.txt', 'a') as f:
            f.write(s + "\n")
        print(s)


# -------------------------------- PORT STUFF --------------------------------------

wport = 1E-5

port_order = 1
port_qpts = 256
port_coil_distance_weight = Weight(2e7)
port_arc_penalty_weight = Weight(2e-3)
port_forward_facing_weight = Weight(2e1)

phic0 = 0.0
phic1 = 0.012
thetac0 = 0.0
thetas1 = 0.1


#######################################################
# End of input parameters.
#######################################################

# Initialize the boundary magnetic surface:
nphi = 200
ntheta = 32
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
fulls = SurfaceRZFourier.from_vmec_input(filename, nphi=4*nphi, ntheta=ntheta)
v = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
v.extend_via_normal(0.132) # when scaled by 1.137438381277359e+01, this is 1.5m from the plasma

# Create the initial coils:
base_tf_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_tf_currents = [Current(1e5) for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
#print(base_currents[0].x)
#base_currents[0].fix_all()
#print(base_currents[0].x)



tf_coils = coils_via_symmetries(base_tf_curves, base_tf_currents, s.nfp, True)
tf_curves = [c.curve for c in tf_coils]

coils = tf_coils
curves = [c.curve for c in coils]

# ---------------------------------------------------------------
#                 GENERATE PORT
c2d = Curve2D( port_qpts, port_order )
port_curve = CurveCWSFourier(
    c2d,
    s
)
s.fix_all()
c2d.unfix_all()

port_curve.curve2d.set('phic(0)', phic0)
port_curve.curve2d.set('phic(1)', phic1)
port_curve.curve2d.set('thetac(0)', thetac0)
port_curve.curve2d.set('thetas(1)', thetas1)

all_ports = apply_symmetries_to_curves( [port_curve], s.nfp, False )
curves_to_vtk( all_ports, OUT_DIR + "ports_initial")

# ----------------------------------------------------------------------------------------------------
#                                       BIOTSAVART OBJECT
# ====================================================================================================

bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))
logprint(f'{bs._coils[0]._current.x}')
logprint(f'{bs._coils[1]._current.x}')
logprint(f'{bs._coils[2]._current.x}')
logprint(f'{bs._coils[3]._current.x}')

coils_to_vtk( bs.coils, OUT_DIR + 'coils_initial' )
surf_to_vtk( OUT_DIR + 'surface_initial', bs, fulls )



# =======================================================================
#                    OBJECTIVE FUNCTION DEFINITION
# -----------------------------------------------------------------------

# Define the individual terms objective function:
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_tf_curves]
Jccdist = CurveCurveDistance(tf_curves, CC_THRESHOLD, num_basecurves=ncoils)
Jcsdist = CurveSurfaceDistance(base_tf_curves, s, CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_tf_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_tf_curves]
linkNum = LinkingNumber(tf_curves)
J_summscs = sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs)
Jarc = sum([ArclengthVariation(c) for c in base_tf_curves])

# Port
Jxyarea = ProjectedEnclosedArea( port_curve, projection='zphi' )
Jccxydist = ProjectedCurveCurveDistance( tf_curves, port_curve, 0.1, projection='zphi' )
Jarcport = ArclengthVariation( port_curve )
Jufp = DirectedFacingPort( port_curve, projection='r')


Jport = -1*Jxyarea \
    + port_coil_distance_weight * Jccxydist \
    + port_arc_penalty_weight * Jarcport \
    + port_forward_facing_weight * Jufp



# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
#
JF =  Jf \
    + LENGTH_WEIGHT * sum([QuadraticPenalty(CurveLength(c), coil_length, 'max') for c in base_tf_curves]) \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT *  J_summscs\
    + linkNum \
    + wport * Jport
    #+ ARCLENGTH_WEIGHT * Jarc
#\
    #

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize


def fun(dofs):
    JF.x = dofs
    bs.set_points(s.gamma().reshape((-1,3)))
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    MaxBdotN = np.max(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    mean_AbsB = np.mean(bs.AbsB())
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_tf_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    outstr += f", ⟨B·n⟩/|B|={BdotN/mean_AbsB:.1e}"
    outstr += f", (Max B·n)/|B|={MaxBdotN/mean_AbsB:.1e}"
    outstr += f", Link Number = {linkNum.J()}"
    logprint(outstr)
    return J, grad


logprint("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
f = fun
dofs = JF.x
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    logprint(f"err = {(J1-J2)/(2*eps) - dJh}")

logprint("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
coils_to_vtk( bs.coils, OUT_DIR + 'coils_tmp' )
curves_to_vtk( all_ports, OUT_DIR + "ports_tmp")
surf_to_vtk( OUT_DIR + 'surface_tmp', bs, fulls )

logprint(res.message)


logprint("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
f = fun
dofs = JF.x
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    logprint(f"err = {(J1-J2)/(2*eps) - dJh}")

# We now use the result from the optimization as the initial guess for a
# subsequent optimization with reduced penalty for the coil length. This will
# result in slightly longer coils but smaller `B·n` on the surface.
wport *= 1e3
JF = QFWeight * QuadraticPenalty(Jf, 1E-5, 'max') \
    + LENGTH_WEIGHT * QuadraticPenalty(sum(Jls), 2.6*ncoils) \
    + CC_WEIGHT * Jccdist \
    + CS_WEIGHT * Jcsdist \
    + CURVATURE_WEIGHT * sum(Jcs) \
    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs) \
    + wport * Jport

dofs = res.x
logprint("""################################################################################
### Second Half #########################################################################
################################################################################""")
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
coils_to_vtk( bs.coils, OUT_DIR + 'coils_final' )
curves_to_vtk( all_ports, OUT_DIR + "ports_final")
surf_to_vtk( OUT_DIR + 'surface_final', bs, fulls )

timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
#This can be rewritten in newer versions of simsopt by saving the BS object as a json file
np.save(OUT_DIR + f'four_field_optimized_{timestamp}', JF.x)

bs.save(OUT_DIR + 'biotsavart.json')

bs.set_points(s.gamma().reshape((-1,3)))
qfl = Jf.J()
jl = sum(Jls).J()
ccdist = Jccdist.shortest_distance()
csdist = Jcsdist.shortest_distance()
maxc = [np.max(np.abs(c.kappa())) for c in base_tf_curves]
msc  = [jj.J() for jj in Jmscs]
summscs = J_summscs.J()
LINKNUM = linkNum.J()
logprint(f'Quadratic flux={qfl:.2E}, Total length={jl:.2E}, ccdist min={ccdist:.2E}, csdist min={csdist:.2E}, max maxc={np.max(maxc):.2E}, summscs={summscs:.2E}, jln={LINKNUM:.2E}\n')

# Port
portarea = Jxyarea.J()
portcurve_mindist = Jccxydist.shortest_distance()
portarc = Jarcport.J()
portufp = Jufp.J()
logprint(f'Port area={portarea:.2E}, Port-curve min distance={portcurve_mindist:.2E}, Port arc penalty={portarc:.2E}, Port UFP penalty={portufp:.2E}.\n')


