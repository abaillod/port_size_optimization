#!/usr/bin/env python3

import os
import numpy as np
from scipy.optimize import minimize

from simsopt.configs import get_ncsx_data
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.geo import SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual, \
    Volume, MajorRadius, CurveLength, NonQuasiSymmetricRatio, Iotas, SurfaceRZFourier, Volume, BoozerResidual
from simsopt.objectives import QuadraticPenalty
from simsopt.util import in_github_actions
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves
from simsopt.field.coil import Current, coils_via_symmetries, ScaledCurrent, apply_symmetries_to_curves


from simsopt.geo import CurveCWSFourierFree
from simsopt.geo import Curve2D
from simsopt.geo import ProjectedEnclosedArea, ProjectedCurveCurveDistance, ProjectedCurveConvexity, DirectedFacingPort

# Port input
with_port = False
PORT_THRESHOLD = 
wport = 1E-4
wdd = Weight( 1E-2 )
warc = Weight( 1E-4 )
wufp = Weight( 1E-4 )


# Directory for output
OUT_DIR = "./boozer_output/"
os.makedirs(OUT_DIR, exist_ok=True)

print("Running 2_Intermediate/boozerQA.py")
print("================================")

surf = SurfaceRZFourier.from_vmec_input('input.nfp4_QH_warm_start')
ncoils = 3
nmodes_coils = 7
R0 = 1.0
R1 = 0.6
base_curves = create_equally_spaced_curves(ncoils, surf.nfp, stellsym=True, R0=R0, R1=R1, order=nmodes_coils, numquadpoints=128)
base_currents = [Current(1) * 1e5 for _ in range(ncoils)]

coils = coils_via_symmetries(base_curves, base_currents, 3, True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)
bs.set_points(surf.gamma().reshape((-1,3)))
bs_tf = BiotSavart(coils)
current_sum = sum(abs(c.current.get_value()) for c in coils)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

## COMPUTE THE INITIAL SURFACE ON WHICH WE WANT TO OPTIMIZE FOR QA##
# Resolution details of surface on which we optimize for qa
mpol = 6  
ntor = 6  
stellsym = True
nfp = surf.nfp

constraint_weight = 1

phis = surf.quadpoints_phi
thetas = surf.quadpoints_theta
s = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)

# #Define one port
# port_order = 2
# port_qpts = 64

# c2dport = Curve2D(port_qpts, port_order )
# c2dport.name = 'Port2DCurve'

# port_curve = CurveCWSFourier(
#     c2dport,
#     s
# )
# port_curve.name = 'PortCurve'

# nfp = s.nfp
# dphi = 1/(2*nfp) * 1/ncoils
# iphi0 = 0
# port_curve.curve2d.set('phic(0)', iphi0*dphi)
# port_curve.curve2d.set('phic(1)', dphi/3.0)
# port_curve.curve2d.set('thetac(0)', 0.0)
# port_curve.curve2d.set('thetas(1)', 0.05)

# ports = apply_symmetries_to_curves( [port_curve], s.nfp, True)


# To generate an initial guess for the surface computation, start with the magnetic axis and extrude outward
s.least_squares_fit( surf.gamma() )
iota = -1.11

# Use a volume surface label
vol = Volume( s )
vol_target = surf.volume()

## compute the surface
boozer_surface = BoozerSurface(bs, s, vol, vol_target, constraint_weight)
res = boozer_surface.run_code('ls', iota, G0, verbose=True)

out_res = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)[0]
print(f"NEWTON {res['success']}: iter={res['iter']}, iota={res['iota']:.3f}, vol={s.volume():.3f}, ||residual||={np.linalg.norm(out_res):.3e}")


## SET UP THE OPTIMIZATION PROBLEM AS A SUM OF OPTIMIZABLES ##
bs_nonQS = BiotSavart(coils)
mr = MajorRadius(boozer_surface)
ls = [CurveLength(c) for c in base_curves]

J_major_radius = QuadraticPenalty(mr, mr.J(), 'identity')  # target major radius is that computed on the initial surface
J_iotas = QuadraticPenalty(Iotas(boozer_surface), res['iota'], 'identity')  # target rotational transform is that computed on the initial surface
J_nonQSRatio = NonQuasiSymmetricRatio(boozer_surface, bs_nonQS)
Jls = QuadraticPenalty(sum(ls), float(sum(ls).J()), 'max') 
JBoozerResidual = BoozerResidual(boozer_surface, bs_nonQS)

# sum the objectives together
JF = J_nonQSRatio + JBoozerResidual + J_iotas + J_major_radius + Jls

curves_to_vtk(curves, OUT_DIR + "curves_init")
boozer_surface.surface.to_vtk(OUT_DIR + "surf_init")

# let's fix the coil current
base_currents[0].fix_all()


def fun(dofs):
    # save these as a backup in case the boozer surface Newton solve fails
    sdofs_prev = boozer_surface.surface.x
    iota_prev = boozer_surface.res['iota']
    G_prev = boozer_surface.res['G']

    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()

    if not boozer_surface.res['success']:
        # failed, so reset back to previous surface and return a large value
        # of the objective.  The purpose is to trigger the line search to reduce
        # the step size.
        J = 1e3
        boozer_surface.surface.x = sdofs_prev
        boozer_surface.res['iota'] = iota_prev
        boozer_surface.res['G'] = G_prev

    cl_string = ", ".join([f"{J.J():.1f}" for J in ls])
    outstr = f"J={J:.1e}, J_nonQSRatio={J_nonQSRatio.J():.2e}, iota={boozer_surface.res['iota']:.2e}, mr={mr.J():.2e}"
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in ls):.1f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad


print("""
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
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
    J1, _ = f(dofs + 2*eps*h)
    J2, _ = f(dofs + eps*h)
    J3, _ = f(dofs - eps*h)
    J4, _ = f(dofs - 2*eps*h)
    print("err", ((J1*(-1/12) + J2*(8/12) + J3*(-8/12) + J4*(1/12))/eps - dJh)/np.linalg.norm(dJh))

print("""
################################################################################
### Run the optimization #######################################################
################################################################################
""")
# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 1e3

res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': MAXITER}, tol=1e-15)
curves_to_vtk(curves, OUT_DIR + "curves_opt")
boozer_surface.surface.to_vtk(OUT_DIR + "surf_opt")

print("End of 2_Intermediate/boozerQA.py")
print("================================")
