import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from simsopt.geo import SurfaceRZFourier
from simsopt._core import load
from simsopt.field import SurfaceClassifier, \
    particles_to_vtk, compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data
import pickle

# Read command line arguments
parser = argparse.ArgumentParser()

# If ran with "--pickle", expect the input to be a pickle.
parser.add_argument("--path", dest="path", default=None)

# Prepare args
args = parser.parse_args()


OUTDIR = args.path
figure_path = os.path.join(OUTDIR, 'figures')
os.makedirs(figure_path, exist_ok=True)

#Load input and output
nphi=128
ntheta=64
surf = SurfaceRZFourier.from_vmec_input('input.LandremanPaul2021_QH', range="half period", nphi=nphi, ntheta=ntheta)

bs_initial = load(os.path.join(OUTDIR, 'biotsavart_initial.json'))
bs_final   = load(os.path.join(OUTDIR, 'biotsavart_final.json')  )
port_initial = load(os.path.join(OUTDIR, 'port_initial.json'))
port_final = load(os.path.join(OUTDIR, 'port_final.json'))

ncoils=5


# =====================================================================
# 2D PLOT OF CURVES, PROJECTED ON XY PLANE
xx = surf.gamma().reshape((-1,3))[:,0]
yy = surf.gamma().reshape((-1,3))[:,1]


def project(x, x0):
    phic = np.arctan2(x0[1], x0[0])
    unit_normal = np.array([np.cos(phic), np.sin(phic), np.zeros(phic.shape)])
    unit_tangent = np.array([-np.sin(phic), np.cos(phic), np.zeros(phic.shape)])
    unit_z = np.array([np.zeros(phic.shape), np.zeros(phic.shape), np.ones(phic.shape)])

    M = np.array([unit_normal,unit_tangent,unit_z]).transpose()
    invM = np.linalg.inv(M)
    
    return np.einsum('ij,...j->...i',invM,x-x0)



## INITIAL CURVES
x0 = np.mean(port_initial.gamma(),axis=0)
gproj = project(surf.gamma().reshape((-1,3)), x0)

fig, ax = plt.subplots()
ax.scatter(gproj[:,1],gproj[:,2])

gport = project(port_initial.gamma(), x0)
ax.fill(gport[:,1], gport[:,2], color='r', alpha=0.7)

for c in bs_initial.coils:
    g = project(c.curve.gamma(), x0)
    zcurves = g[:,0]

    ind = np.where( zcurves>0 )[0]
    g = g[ind,:]

    ax.scatter(g[:,1], g[:,2], color='k', marker='o', s=15)

ax.set_aspect('equal')
ax.set_xlabel('\phi')
ax.set_ylabel('z')
plt.savefig(os.path.join(figure_path, 'port_size_horizontal_view_initial'))

## FINAL CURVES
x0 = np.mean(port_final.gamma(),axis=0)
gproj = project(surf.gamma().reshape((-1,3)), x0)

fig, ax = plt.subplots()
ax.scatter(gproj[:,1],gproj[:,2])

gport = project(port_final.gamma(), x0)
ax.fill(gport[:,1], gport[:,2], color='r', alpha=0.7)

for c in bs_final.coils:
    g = project(c.curve.gamma(), x0)
    zcurves = g[:,0]

    ind = np.where( zcurves>0 )[0]
    g = g[ind,:]

    ax.scatter(g[:,1], g[:,2], color='k', marker='o', s=15)

ax.set_aspect('equal')
ax.set_xlabel('\phi')
ax.set_ylabel('z')
plt.savefig(os.path.join(figure_path, 'port_size_horizontal_view_final'))

# =====================================================================
# NORMAL FIELD ERROR

## INITIAL
theta = surf.quadpoints_theta
phi = surf.quadpoints_phi
ntheta = theta.size
nphi = phi.size
bs_initial.set_points(surf.gamma().reshape((-1,3)))
Bdotn = np.sum(bs_initial.B().reshape((nphi, ntheta, 3)) * surf.unitnormal(), axis=2)
modB = bs_initial.AbsB().reshape((nphi,ntheta))

fig, ax = plt.subplots(figsize=(12,5))
c = ax.contourf(theta,phi,Bdotn / modB)
plt.colorbar(c)
ax.set_title(r'$\mathbf{B}\cdot\hat{n} / |B|$ ')
ax.set_ylabel(r'$\phi$')
ax.set_xlabel(r'$\theta$')
plt.savefig(os.path.join(figure_path, 'normal_field_error_initial.png'))


## FINAL
theta = surf.quadpoints_theta
phi = surf.quadpoints_phi
ntheta = theta.size
nphi = phi.size
bs_final.set_points(surf.gamma().reshape((-1,3)))
Bdotn = np.sum(bs_final.B().reshape((nphi, ntheta, 3)) * surf.unitnormal(), axis=2)
modB = bs_final.AbsB().reshape((nphi,ntheta))

fig, ax = plt.subplots(figsize=(12,5))
c = ax.contourf(theta,phi,Bdotn / modB)
plt.colorbar(c)
ax.set_title(r'$\mathbf{B}\cdot\hat{n} / |B|$ ')
ax.set_ylabel(r'$\phi$')
ax.set_xlabel(r'$\theta$')
plt.savefig(os.path.join(figure_path, 'normal_field_error_final.png'))




# =====================================================================
# POINCARE

# Run and plot Poincare section
bs = bs_final
surf = SurfaceRZFourier.from_vmec_input('input.LandremanPaul2021_QH', range="half period", nphi=nphi, ntheta=ntheta)
surf1 = SurfaceRZFourier.from_vmec_input('input.LandremanPaul2021_QH', range="half period", nphi=nphi, ntheta=ntheta)
surf1.extend_via_normal(0.1)
nfp = surf.nfp

Rmaj = surf.major_radius()
r0 = surf.minor_radius()
sc_fieldline = SurfaceClassifier(surf1, h=0.01, p=3)
nfieldlines = 50
tmax_fl = 2500
degree = 4

def trace_fieldlines(bfield,label):
    # Set up initial conditions - 
    R0 = np.linspace(Rmaj-2*r0, Rmaj+2*r0, nfieldlines)
    Z0 = np.zeros(nfieldlines)
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-8,
        phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
    plot_poincare_data(fieldlines_phi_hits, phis, os.path.join(figure_path, 'poincare_final'), dpi=150,surf=surf,mark_lost=True)
    return fieldlines_phi_hits

hits = trace_fieldlines(bs_final, 'vmec')


