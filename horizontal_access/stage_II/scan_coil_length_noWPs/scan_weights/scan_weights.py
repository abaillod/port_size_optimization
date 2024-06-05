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
import json
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

ncoils = 5
port_qpts = 256
phic0 = 0.0
phic1 = 0.012
thetac0 = 0.0
thetas1 = 0.1

R0 = 1.0

# Number of iterations to perform:
MAXITER = 500

# File for the desired boundary magnetic surface:
filename = 'input.LandremanPaul2021_QH'

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)


# Initialize the boundary magnetic surface:
nphi = 200
ntheta = 32
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
fulls = SurfaceRZFourier.from_vmec_input(filename, nphi=4*nphi, ntheta=ntheta)
v = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
v.extend_via_normal(0.132) # when scaled by 1.137438381277359e+01, this is 1.5m from the plasma




#######################################################
# End of input parameters.
#######################################################

def run_optimization(
    R1, order, port_order, LENGTH_WEIGHT, coil_length, CC_THRESHOLD,
    CC_WEIGHT, CS_WEIGHT, CS_THRESHOLD, CURVATURE_THRESHOLD, CURVATURE_WEIGHT, MSC_THRESHOLD,
    MSC_WEIGHT, LNUM_WEIGHT, ARCLENGTH_WEIGHT, port_coil_distance_weight, port_arc_penalty_weight,
    port_forward_facing_weight
):

    directory = (
        f"ncoils_{ncoils}_order_{order}_R1_{R1:.2}_l_{coil_length:.2}_w_{LENGTH_WEIGHT.value:.2}"
        + f"_mc_{CURVATURE_THRESHOLD:.2}_w_{CURVATURE_WEIGHT.value:.2}"
        + f"_msc_{MSC_THRESHOLD:.2}_w_{MSC_WEIGHT.value:.2}"
        + f"_cc_{CC_THRESHOLD:.2}_w_{CC_WEIGHT.value:.2}"
        + f"_cs_{CS_THRESHOLD:.2}_w_{CS_WEIGHT.value:.2}"
        + f"_lnw_{LNUM_WEIGHT.value:.2}"
        + f"alw_{ARCLENGTH_WEIGHT.value:.2}_po_{port_order}_paw_{wport.value:.2}"
        + f"_pcdw_{port_coil_distance_weight.value:.2}_palw_{port_arc_penalty_weight.value:.2}"
        + f"_pffw_{port_forward_facing_weight.value:.2}"
    )

    new_OUT_DIR = os.path.join(OUT_DIR,  directory + "/")
    os.mkdir(new_OUT_DIR)

    logfile = os.path.join(OUT_DIR, directory, 'log.txt')
    with open(logfile, 'w') as f:
        f.write('HORIZONTAL PORT SIZE OPTIMIZATION\n')
        
    def logprint(s):
        if comm_world.rank == 0:
            with open(logfile, 'a') as f:
                f.write(s + "\n")
            print(s)
    
    # Create the initial coils:
    base_tf_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
    base_tf_currents = [Current(1e5) for i in range(ncoils)]    
    
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
    #curves_to_vtk( all_ports, OUT_DIR + "ports_initial")
    
    # ----------------------------------------------------------------------------------------------------
    #                                       BIOTSAVART OBJECT    
    bs = BiotSavart(coils)
    bs.set_points(s.gamma().reshape((-1, 3)))
    
    #coils_to_vtk( bs.coils, OUT_DIR + 'coils_initial' )
    #surf_to_vtk( OUT_DIR + 'surface_initial', bs, fulls )
    
    
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
    
    
    Jport = -1 * wport.value*Jxyarea \
        + port_coil_distance_weight * Jccxydist \
        + port_arc_penalty_weight * Jarcport \
        + port_forward_facing_weight * Jufp
    
    JF =  Jf \
        + LENGTH_WEIGHT * sum([QuadraticPenalty(CurveLength(c), coil_length, 'max') for c in base_tf_curves]) \
        + CC_WEIGHT * Jccdist \
        + CS_WEIGHT * Jcsdist \
        + CURVATURE_WEIGHT * sum(Jcs) \
        + MSC_WEIGHT *  J_summscs\
        + linkNum \
        + wport * Jport

    iteration=0
    def fun(dofs):
        nonlocal iteration
        iteration+=1
        JF.x = dofs
        bs.set_points(s.gamma().reshape((-1,3)))
        J = JF.J()
        grad = JF.dJ()
        jf = Jf.J()
        BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        MaxBdotN = np.max(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
        mean_AbsB = np.mean(bs.AbsB())
        outstr = f"Iteration {iteration:4} :: J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
        cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_tf_curves)
        msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
        outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        outstr += f", ⟨B·n⟩/|B|={BdotN/mean_AbsB:.1e}"
        outstr += f", (Max B·n)/|B|={MaxBdotN/mean_AbsB:.1e}"
        outstr += f", Link Number = {linkNum.J()}"
        print(outstr)
        return J, grad

    s.fix_all()
    fulls.fix_all()
    v.fix_all()
    for c in coils:
        c.unfix_all()
    coils[0].current.fix_all()

    for c in JF.dof_names:
        print(c)

    dofs = JF.x

    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
    JF.x = res.x
    logprint(res.message)
    bs.set_points(s.gamma().reshape((-1, 3)))

    bs.save(new_OUT_DIR + "biotsavart.json")
    Bbs = bs.B().reshape((nphi, ntheta, 3))
    BdotN = np.max(np.sum(Bbs * s.unitnormal(),axis=2) / np.linalg.norm(Bbs, axis=2))


    results = {
        "R0": R0,
        "R1": R1,
        "order": order,
        "nphi": nphi,
        "ntheta": ntheta,
        "length_weight": LENGTH_WEIGHT.value,
        "length_target": coil_length,
        "max_curvature_weight": CURVATURE_WEIGHT.value,
        "max_curvature_threshold": CURVATURE_THRESHOLD,
        "cc_weight": CC_WEIGHT.value,
        "cc_threshold": CC_THRESHOLD,
        "cs_weight": CS_WEIGHT.value,
        "cs_threshold": CS_THRESHOLD,
        "mean_squared_curvature_threshold": MSC_THRESHOLD,
        "mean_squared_curvature_weight": MSC_WEIGHT.value,
        "linking_number_weight": LNUM_WEIGHT.value,
        "arclength_weight": ARCLENGTH_WEIGHT.value,
        "port_area_weight": wport.value,
        "port_coil_distance_weight": port_coil_distance_weight.value,
        "port_arc_penalty_weight": port_arc_penalty_weight.value,
        "port_forward_facing_weight": port_forward_facing_weight.value,
        "message": res.message,
        "success": res.success,
        "iterations": res.nit,
        "function_evaluations": res.nfev,
        
        "JF": float(JF.J()),
        "Jf": float(Jf.J()),
        "BdotN": BdotN,
        "lengths": [float(J.J()) for J in Jls],
        "length": float(sum(J.J() for J in Jls)),
        "average_length_per_coil": float(sum(J.J() for J in Jls))/ncoils,
        "max_curvatures": [np.max(c.kappa()) for c in base_tf_curves],
        "max_max_curvature": max(np.max(c.kappa()) for c in base_tf_curves),
        "coil_coil_distance": Jccdist.shortest_distance(),
        "gradient_norm": np.linalg.norm(JF.dJ()),
        "linking_number": LinkingNumber(curves).J(),
        "coil_surface_distance":  float(Jcsdist.shortest_distance()),
        "port_area": float(Jxyarea.J()),
        "port_coil_distance": float(Jccxydist.shortest_distance()),
        "port_direction_penalty": float(Jufp.J())       
    }
    

    with open(new_OUT_DIR + "results.json", "w") as outfile:
        json.dump(results, outfile, indent=2)



def rand(min, max):
    """Generate a random float between min and max."""
    return np.random.rand() * (max - min) + min

for ii in range(2):
    R1 = rand(0.45, 0.55)

    order = int(np.round(rand(5, 12)))
    
    port_order = int(np.round(rand(1, 5)))
    
    LENGTH_WEIGHT = Weight( 10**rand(-3, 0) ) 
    coil_length = rand(2.5, 4.0)
    
    CC_THRESHOLD = rand(0.05, 0.15)
    CC_WEIGHT = Weight( 10**rand(1, 3) )
    
    CS_THRESHOLD = rand(0.1, 0.2)
    CS_WEIGHT = Weight( 10**rand(2,4) )
    
    CURVATURE_THRESHOLD = rand(8,15)
    CURVATURE_WEIGHT = Weight( 10**rand(-8,-5) )
    
    MSC_THRESHOLD = rand(6,15)
    MSC_WEIGHT = Weight( 10**rand(-9,-6) )
    
    LNUM_WEIGHT = Weight( 10**rand(-2,2) )
    
    ARCLENGTH_WEIGHT = Weight( 10**rand(-4,2) )
    
    wport = Weight( 10**rand(-7,2) )
    
    port_coil_distance_weight = Weight( 10**rand(0,6) )
    port_arc_penalty_weight = Weight( 10**rand(-4,2) )
    port_forward_facing_weight = Weight( 10**rand(-2,4) )

    run_optimization(
        R1, order, port_order, LENGTH_WEIGHT, coil_length, CC_THRESHOLD,
        CC_WEIGHT, CS_WEIGHT, CS_THRESHOLD, CURVATURE_THRESHOLD, CURVATURE_WEIGHT, MSC_THRESHOLD,
        MSC_WEIGHT, LNUM_WEIGHT, ARCLENGTH_WEIGHT, port_coil_distance_weight, port_arc_penalty_weight,
        port_forward_facing_weight
    )