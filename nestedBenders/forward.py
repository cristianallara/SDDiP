__author__ = "Cristiana L. Lara"

from pyomo.environ import *


def forward_pass(t, bl, time_periods, rn_r, th_r):

    # Solve the model
    mipsolver = SolverFactory('gurobi')
    mipsolver.options['mipgap'] = 0.0001
    mipsolver.options['timelimit'] = 40
    # mipsolver.options['threads'] = 6
    mipsolver.solve(bl)  # , tee=True)#,save_results=False)    m.Bl[t, n]

    ngo_rn_par = {}
    ngo_th_par = {}

    # Fix the linking variable as parameter for next t
    if t != time_periods:
        for (rn, r) in rn_r:
            ngo_rn_par[rn, r] = bl.ngo_rn[rn, r].value

        for (th, r) in th_r:
            ngo_th_par[th, r] = bl.ngo_th[th, r].value

    # Store obj value to compute UB
    cost = bl.obj() - bl.alphafut.value

    return ngo_rn_par, ngo_th_par, cost
