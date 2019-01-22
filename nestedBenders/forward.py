__author__ = "Cristiana L. Lara"

from pyomo.environ import *


def forward_pass(bl, state_vars, opt_tol, time_limit):

    # Solve the model
    mipsolver = SolverFactory('gurobi')
    mipsolver.options['mipgap'] = opt_tol
    mipsolver.options['timelimit'] = time_limit
    # mipsolver.options['threads'] = 6
    mipsolver.solve(bl)  # , tee=True)#,save_results=False)    m.Bl[t, n]

    state_vars_val = [{} for i in range(len(state_vars))]

    # Fix the state variable as parameter for next t
    for i in range(len(state_vars)):
        for key in state_vars[i].keys():  
            state_vars_val[i][key] = state_vars[i][key].value

    # Store obj value to compute UB
    cost = bl.obj() - bl.alphafut.value

    return state_vars_val, cost
