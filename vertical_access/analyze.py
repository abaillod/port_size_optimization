import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from simsopt.geo import SurfaceRZFourier
from simsopt._core import load
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

#bs_initial = load(os.path.join(OUTDIR, 'biotsavart_initial.json'))
bs_final   = load(os.path.join(OUTDIR, 'biotsavart_final.json')  )
port_initial = load(os.path.join(OUTDIR, 'port_initial.json'))
port_final = load(os.path.join(OUTDIR, 'port_final.json'))

ncoils=5


# =====================================================================
# 2D PLOT OF CURVES, PROJECTED ON XY PLANE
xx = surf.gamma().reshape((-1,3))[:,0]
yy = surf.gamma().reshape((-1,3))[:,1]

## INITIAL CURVES
# fig, ax = plt.subplots()
# ax.scatter(xx,yy,s=1)

# g = port_initial.gamma()
# ax.fill(g[:,0], g[:,1], color='r', alpha=0.8)

# for c in bs_initial.coils[:ncoils]:
#     g = c.curve.gamma()
#     ind = np.where(g[:,2]>0)
#     ax.scatter(g[ind,0], g[ind,1], color='k', marker='*', s=5)

# ax.set_aspect('equal')
# ax.set_xlabel('x')
# ax.set_ylabel('y')

# plt.savefig(os.path.join(figure_path, 'port_size_vertical_view_initial'))

## FINAL CURVES
fig, ax = plt.subplots()
ax.scatter(xx,yy,s=1)

g = port_final.gamma()
ax.fill(g[:,0], g[:,1], color='r', alpha=0.8)

for c in bs_final.coils[:ncoils]:
    g = c.curve.gamma()
    ind = np.where(g[:,2]>0)
    ax.scatter(g[ind,0], g[ind,1], color='k', marker='*', s=5)

ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.savefig(os.path.join(figure_path, 'port_size_vertical_view_final'))


# =====================================================================
# NORMAL FIELD ERROR

## INITIAL
# theta = surf.quadpoints_theta
# phi = surf.quadpoints_phi
# ntheta = theta.size
# nphi = phi.size
# bs_initial.set_points(surf.gamma().reshape((-1,3)))
# Bdotn = np.sum(bs_initial.B().reshape((nphi, ntheta, 3)) * surf.unitnormal(), axis=2)
# modB = bs_initial.AbsB().reshape((nphi,ntheta))

# fig, ax = plt.subplots(figsize=(12,5))
# c = ax.contourf(theta,phi,Bdotn / modB)
# plt.colorbar(c)
# ax.set_title(r'$\mathbf{B}\cdot\hat{n} / |B|$ ')
# ax.set_ylabel(r'$\phi$')
# ax.set_xlabel(r'$\theta$')
# plt.savefig(os.path.join(figure_path, 'normal_field_error_initial.png'))


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


