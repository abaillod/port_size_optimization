""" Maximize port size

This script optimize the port size of a configuration while 
keeping the field error, total coil length, coil mean and max curvature,
coil-coil and coil-surface distance below a given threshold
"""
import numpy as np
from simsopt.geo import PortSize
from simsopt._core.optimizable import load, Optimizable
from simsopt.mhd import Vmec
from simsopt.geo import SurfaceRZFourier
from jax import grad
from simsopt.objectives import SquaredFlux, QuadraticPenalty
from simsopt.geo import CurveLength, CurveCurveDistance, MeanSquaredCurvature, LpCurveCurvature
from simsopt.geo import curves_to_vtk, create_equally_spaced_curves, CurveSurfaceDistance, ToroidalFlux
import os
from simsopt.field.coil import apply_symmetries_to_currents, apply_symmetries_to_curves, ScaledCurrent
from simsopt.field import BiotSavart, Current
from simsopt.field.coil import Coil
from pystellplot.Paraview import coils_to_vtk, surf_to_vtk
from scipy.optimize import minimize
import datetime

# Scale factor
alpha = 11.37

# User inputs
CC_dist = alpha*0.08 # based on 83cm winding pack size + 20% margin
wcc = 156.434465
CS_dist = alpha*0.12 # based on blanket, structure, vessel, LT shield, gaps
wcs = 1564.34465
wl = 0.0156434465
wport = 1e-5
fpsin = 5
kth = 12.
wk = 0.000000156434465

now = datetime.datetime.now()
fname = f"{now.strftime('%Y%m%d')}/wiedman_warm_start_wport={wport:.1E}_fpsin={fpsin:.1E}" 
mydir = os.path.join('../runs', fname)
os.makedirs(now.strftime('%Y%m%d'), exist_ok=True)
os.makedirs(mydir, exist_ok=True)
coil_path = os.path.join(mydir, 'coils')
os.makedirs(coil_path)
surf_path = os.path.join(mydir, 'surf')
os.makedirs(surf_path)

# Read stuff
v = Vmec('input.scaled')
nfp = v.boundary.nfp
surf = v.boundary
nfp = v.boundary.nfp    

# Initial coils
base_curves = create_equally_spaced_curves( ncoils, nfp, True, R0=v.boundary.major_radius(), R1=v.boundary.minor_radius()*3, order=10 )
full_curves = apply_symmetries_to_curves( base_curves, nfp, True )

base_currents = [Current(1) for c in base_curves]
full_currents = [ScaledCurrent(c, 1e5) for c in apply_symmetries_to_currents( base_currents, nfp, True )]

coils = [Coil(curve, current) for curve, current in zip(full_curves, full_currents)] #+ [hcoil]
bs = BiotSavart( coils )
bs2 = BiotSavart( coils )
bs.set_points(surf.gamma().reshape((-1,3)))

coils_to_vtk( coils, os.path.join(coil_path, f'coils.ncoil={ncoils}_nwp={nwp}.initial.json' ) )
surf_to_vtk( os.path.join(surf_path, f'surf.ncoil={ncoils}_nwp={nwp}.initial.json'), bs, surf )
bs.save( os.path.join(coil_path, f'biotsavart.ncoil={ncoils}_nwp={nwp}.initial.json'), fmt="json" )

logfile = os.path.join(mydir, 'logging.out')
with open(logfile, 'w') as f:
    f.write(f'Optimization of port size, with ncoils={ncoils}, nwp={nwp}\n')
def log_print(mystr):
    with open(logfile, 'a') as f:
        f.write(mystr)



# First optimize for low B dot n
sqflux = SquaredFlux( surf, bs )
torflux = QuadraticPenalty( ToroidalFlux( surf, bs2 ), tflux, 'identity' )
lengths = QuadraticPenalty( sum([CurveLength(curve) for curve in base_curves]), 2.6*alpha*len(base_curves), 'max' )
cc_penalty = CurveCurveDistance( full_curves, CC_dist, len(base_curves) )
cs_penalty = CurveSurfaceDistance( base_curves, surf, CS_dist )
Jmscs = sum([QuadraticPenalty(MeanSquaredCurvature(c), kth, 'max') for c in base_curves])
linkNum = QuadraticPenalty( LinkingNumber(curves), 0.1, 'max' ) # use quad penalty to deal with numerical errors

JF = sqflux + wl * lengths + wcc*cc_penalty + wcs*cs_penalty + wk * Jmscs + linkNum

log_print(f'Original squared flux penalty is {sqflux.J()}\n')
log_print(f'Max length is {Length}, starting length is {lengths.J()}\n')
log_print(f'Min coil-coil distance is {CC_dist}, starting minimum distance is {cc_penalty.shortest_distance()}\n')
log_print(f'Min plasma-coil distance is {CS_dist}, starting minimum distance is {cs_penalty.shortest_distance()}\n')
log_print(f'Original target function is {JF.J()}\n')

# Define function
def fun(x, info):
    JF.x = x
    info['Nfeval'] += 1
    J = JF.J()
    dJ = JF.dJ()
    if np.mod(info['Nfeval'],10)==0:
        nf = info['Nfeval']
        log_print(f'Evaluation #{nf}: J={J}, max(|dJ|)={np.max(np.abs(dJ))}\n')
    return J, dJ

# Free some dofs and optimize
JF.fix_all()
for current in base_currents:
    current.unfix_all()
base_currents[0].fix_all()
for order in [1, 2, 3, 4]:
    for curve in base_curves:
        for ii in range(0,order):
            curve.unfix(f'xc({ii})')
            curve.unfix(f'yc({ii})')
            curve.unfix(f'zc({ii})')
            if ii>0:
                curve.unfix(f'xs({ii})')
                curve.unfix(f'ys({ii})')
                curve.unfix(f'zs({ii})')
    
    options={'maxiter': 1000, 'maxcor': 300}
    dofs = JF.x
    log_print(f'OPTIMIZATION FOR ORDER={order}\n')
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', args=({'Nfeval':0}), options=options, tol=1e-12)
    log_print(res.message)
    log_print(f'Optimized squared flux penalty is {sqflux.J()}\n')
    log_print(f'Max length is {Length}, optimized length is {lengths.J()}\n')
    log_print(f'Min coil-coil distance is {CC_dist}, optimized minimum distance is {cc_penalty.shortest_distance()}\n')
    log_print(f'Min plasma-coil distance is {CS_dist}, optimized minimum distance is {cs_penalty.shortest_distance()}\n')
    log_print(f'Final target function is {JF.J()}\n')
    log_print(f'============================================\n\n')


    coils_to_vtk( coils, os.path.join(coil_path, f'coils.ncoil={ncoils}_nwp={nwp}.order={order}.json' ) )
    surf_to_vtk( os.path.join(surf_path, f'surf.ncoil={ncoils}_nwp={nwp}.order={order}.json'), bs, surf )
    bs.save( os.path.join(coil_path, f'biotsavart.ncoil={ncoils}_nwp={nwp}.order={order}.json'), fmt="json")
bs.save( os.path.join(coil_path, f'biotsavart.ncoil={ncoils}_nwp={nwp}.post_stage_2.json'), fmt="json")






# ======================================================================================================
# Constrained optimizaiton of port size
nphi = 35
ntheta = 42
three_halves_field_period_surf = SurfaceRZFourier(mpol=v.boundary.mpol,ntor=v.boundary.ntor, nfp=v.boundary.nfp, 
                        quadpoints_phi=np.linspace(0,3/nfp,3*nphi), 
                        quadpoints_theta=np.linspace(0,1,ntheta))   
for mm in range(0,v.boundary.mpol+1):
    for nn in range(-v.boundary.ntor,v.boundary.ntor+1):
        if mm==0 and nn<0:
            continue
        three_halves_field_period_surf.set(f'rc({mm},{nn})', v.boundary.get(f'rc({mm},{nn})'))
        if not (mm==0 and nn==0):
            three_halves_field_period_surf.set(f'zs({mm},{nn})', v.boundary.get(f'zs({mm},{nn})'))
three_halves_field_period_surf.extend_via_normal(0.95*cs_penalty.shortest_distance())
vessel = three_halves_field_period_surf

ncoils = int(len(bs.coils) / (2*nfp) )
three_halves_field_period_curves = full_curves[0:ncoils] + full_curves[2*ncoils:3*ncoils] + full_curves[(2*nfp-1)*ncoils:]
Jport = PortSize(three_halves_field_period_curves, vessel)

# penalties...
sqflux =  SquaredFlux( surf, bs )
max_sq_flux = SquaredFlux.J() * fpsin
Jsqflux = QuadraticPenalty( sqflux, max_sq_flux, 'max' )
JF = -wport*Jport + Jsqflux + wl * lengths + wcc*cc_penalty + wcs*cs_penalty + wk * Jmscs + linkNum

def fun2(x, info):
    JF.x = x
    info['Nfeval'] += 1
    J = JF.J()
    dJ = JF.dJ()
    nf = info['Nfeval']
    log_print(f'Evaluation #{nf}: J={J}, max(|dJ|)={np.max(np.abs(dJ))}.')
    if np.mod(info['Nfeval'],5)==0:
        
        outstr = f"ITERATION {info['Nfeval']}, Jport={-Jport.J():.5E}, sqflux={sqflux.J():.5E}, torflux={torflux.J():.5E}, min CC dist={cc_penalty.shortest_distance():.3E}, min CS dist={cs_penalty.shortest_distance():.3E}\n"
        outstr += f"Lengths = {lengths.J():.3E} "
        outstr += "/n"
        log_print(outstr)

        fname = os.path.join(coil_path, f"biotsavart.ncoil={ncoils}_nwp={nwp}.{info['Nfeval']:04d}.json")
        bs.save( fname,  fmt='json' )
    return J, dJ
    

options={'maxiter': 1000, 'maxcor': 300}
dofs = JF.x
log_print(f'\n================================================================\n')
log_print(f'OPTIMIZATION FOR PORT SIZE\n')
log_print(f'Initial max port size: {Jport.J()}\n')
res = minimize(fun2, dofs, jac=True, method='L-BFGS-B', args=({'Nfeval':0}), options=options, tol=1e-12)
log_print(f'Final max port size: {Jport.J()}\n')
log_print(res.message)
log_print(f'============================================\n\n')


coils_to_vtk( coils, os.path.join(coil_path, f'coils.ncoil={ncoils}_nwp={nwp}.final.json' ) )
surf_to_vtk( os.path.join(surf_path, f'surf.ncoil={ncoils}_nwp={nwp}.final.json'), bs, surf )
bs.save( os.path.join(coil_path, f'biotsavart.ncoil={ncoils}_nwp={nwp}.final.json'), fmt="json")
