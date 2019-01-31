__author__ = "Cristiana L. Lara"

from pyomo.environ import *


def backward_pass(stage, bl, n_stages, rn_r, th_r):
    if stage == n_stages:
        bl.alphafut.fix(0)
    else:
        bl.alphafut.unfix()

    # Solve the model
    opt = SolverFactory('gurobi')
    opt.options['relax_integrality'] = 1
    opt.options['timelimit'] = 40
    # opt.options['threads'] = 6
    opt.solve(bl)  # , tee=True)#, save_results=False)#

    mltp_o_rn = {}
    mltp_o_th = {}

    if stage != 1:
        # Get Lagrange multiplier from linking equality
        for rn_r_index in range(len(rn_r)):
            i = rn_r[rn_r_index][0]
            j = rn_r[rn_r_index][1]
            mltp_o_rn[i, j] = - bl.dual[bl.link_equal1[rn_r_index + 1]]

        for th_r_index in range(len(th_r)):
            i = th_r[th_r_index][0]
            j = th_r[th_r_index][1]
            mltp_o_th[i, j] = - bl.dual[bl.link_equal2[th_r_index + 1]]

    # Get optimal value
    cost = bl.obj()

    return mltp_o_rn, mltp_o_th, cost
