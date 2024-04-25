from simsopt._core import load
from simsopt.geo import SurfaceRZFourier
from simsopt.geo import VerticalPortDiscrete
import os

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from simsopt.geo import CurveCWSFourier, Curve2D
from simsopt.geo import ProjectedEnclosedArea, ProjectedCurveCurveDistance, ProjectedCurveConvexity, DirectedFacingPort
from scipy.optimize import minimize
from simsopt.geo import ArclengthVariation
from simsopt.objectives import Weight

from simsopt.geo.curve import curves_to_vtk
from simsopt.field.coil import apply_symmetries_to_curves
from pystellplot.Paraview import coils_to_vtk, surf_to_vtk

# INPUTS
MAXITER_I = 1E2 # Max number of iteration when iterating on weights
MAXITER_II = 1E4 # Max number of iteration for optimization of port
port_order = 6
port_qpts = 128

curve_port_threshold = 0.15

wdd = Weight(1E-2)  # Start coil-port distance weight
warc = Weight(1E-4) # Start arclength penalty weight
wufp = Weight(1E-2) # Start frontal facing port penalty weight

phic0 = 0.004 #0.035, 0.065, 0.085
phic1 = 0.008
thetac0 = 0.0
thetas1 = 0.05

# Create output directory
os.makedirs('output', exist_ok=True)

# Load the surface, assumed fixed in this notebook. 
surf = SurfaceRZFourier.from_vmec_input( '/Users/antoinebaillod/Projects/Accessibilty/configurations/qh_landreman_paul/input.scaled' )

nfp = surf.nfp
qpts_phi = np.linspace(0, 1/(2*nfp), 16)
qpts_theta = np.linspace(0, 1, 32)

vessel = SurfaceRZFourier(
    nfp = surf.nfp,
    mpol = surf.mpol,
    ntor = surf.ntor,
    stellsym = surf.stellsym,
    quadpoints_phi=qpts_phi,
    quadpoints_theta=qpts_theta,
    dofs = surf.dofs
)

bs = load( '/Users/antoinebaillod/Projects/Accessibilty/configurations/qh_landreman_paul/wiedman_coils/coils.wiedman.json' )
ncoils = 5
curves = [c.curve for c in bs.coils[0:2*ncoils] + bs.coils[-ncoils:]]


order = port_order
qpts = port_qpts
c2d = Curve2D( qpts, order )
curve_cws = CurveCWSFourier(
    c2d,
    vessel
)
vessel.fix_all()

curve_cws.curve2d.set('phic(0)', phic0)
curve_cws.curve2d.set('phic(1)', phic1)
curve_cws.curve2d.set('thetac(0)', thetac0)
curve_cws.curve2d.set('thetas(1)', thetas1)
dofs0 = curve_cws.x


# Define the objective function
bs = load( '/Users/antoinebaillod/Projects/Accessibilty/configurations/qh_landreman_paul/wiedman_coils/coils.wiedman.json' )
ncoils = 5
curves = [c.curve for c in bs.coils[0:2*ncoils] + bs.coils[-ncoils:]]
full_curves = [c.curve for c in bs.coils]



Jxyarea = ProjectedEnclosedArea( curve_cws, projection='zphi' )
Jccxydist = ProjectedCurveCurveDistance( curves, curve_cws, curve_port_threshold, projection='zphi' )
Jconvex = ProjectedCurveConvexity( curve_cws, projection='zphi' )
Jarc = ArclengthVariation( curve_cws )
Jufp = DirectedFacingPort(curve_cws, projection='r')


J = -1*Jxyarea + wdd * Jccxydist + warc * Jarc + wufp*Jufp #+ wco * Jconvex  #+ wph * (Jphimin + Jphimax) + wco * Jconvex # # 

def f(x, info={'Nfeval': 0, 'print': False}):
    info['Nfeval'] += 1
    
    J.x = x

    out = J.J()
    if info['print']:
        print(f"Neval={info['Nfeval']}, J={out:.6E}")    
    return out, J.dJ()

# Save initial curves
ports =  apply_symmetries_to_curves( [curve_cws], surf.nfp, False )
curves_to_vtk(full_curves, 'output/modular_coils')

bs.save('output/biotsavart.json')
curve_cws.save('output/port_initial.json')
curves_to_vtk(ports, 'output/port_initial')
surf_to_vtk( "output/surface", bs, surf )


# Set dofs
for c in curves:
    c.fix_all()
curve_cws.unfix_all()
vessel.fix_all()

dofs = J.x


# Find weights
satisfied = False
counter=1
while not satisfied:
    #curve_cws.x = dofs0
    print(f'Weight iteration {counter}:: Running with wdd={wdd.value:.2E}, warc={warc.value:.2E}, wufp={wufp.value:.2E}')
    MAXITER = MAXITER_I
    dofs = dofs0
    res = minimize(f, dofs, jac=True, args=({'Nfeval': 0, 'print':False}), method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-12)

    satisfied = True

    cpdist = Jccxydist.J()
    arcpen = Jarc.J()
    ufppen = Jufp.J()
    print(f'cpdist={cpdist:.2E}, arcpen={arcpen:.2E}, ufp={ufppen:.2E}')
    if cpdist>1E-6:
        print('cpdist too large')
        wdd *= 1.5
        satisfied = False

    if arcpen>1:
        print('arcpen too large')
        warc *= 1.5
        satisfied = False

    if ufppen>= 0.01:
        print('ufppen too large')
        wufp *= 1.5
        satisfied = False

    counter += 1


MAXITER = MAXITER_II
res = minimize(f, dofs, jac=True, args=({'Nfeval': 0, 'print':True}), method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-12)
print(res.message)


cpdist = Jccxydist.J()
arcpen = Jarc.J()
ufppen = Jufp.J()
if cpdist>1E-6:
    print('cpdist too large')

elif arcpen>1:
    print('arcpen too large')

elif ufppen>= 0.01:
    print('ufppen too large')


curve_cws.save('output/port_final.json')
curves_to_vtk(ports, 'output/port_final')


