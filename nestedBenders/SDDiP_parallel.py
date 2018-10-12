__author__ = "Cristiana L. Lara"
# Stochastic Dual Dynamic Integer Programming (SDDiP) description at:
# https://link.springer.com/content/pdf/10.1007%2Fs10107-018-1249-5.pdf

# This algorithm scenario tree satisfies stage-wise independence

import time
import math
import random
from copy import deepcopy
import os.path
from pyomo.environ import *
import pymp

from scenarioTree import create_scenario_tree
import readData
import optBlocks as b

start_time = time.time()

# ######################################################################################################################
# USER-DEFINED PARAMS

# Define case-study
curPath = os.path.abspath(os.path.curdir)
filepath = os.path.join(curPath, 'GTEPdata_5years.db')
time_periods = 5
stages = range(1, time_periods + 1)
scenarios = ['L', 'R', 'H']
single_prob = {'L': 1/3, 'R': 1/3, 'H': 1/3}

# Define parameters of the decomposition
max_iter = 10
opt_tol = 1  # %
ns = 30  # Number of scenarios solved per Forward/Backward Pass
# NOTE: ns should be between 1 and len(n_stage[time_periods])
z_alpha_2 = 1.96  # 95% confidence level

# ######################################################################################################################

# create scenarios and input data
nodes, n_stage, parent_node, children_node, prob, sc_nodes = create_scenario_tree(stages, scenarios, single_prob)
sc_headers = list(sc_nodes.keys())
readData.read_data(filepath, stages, n_stage)

# create blocks
m = b.create_model(time_periods, max_iter, n_stage, nodes, prob)
print('finished generating the blocks')
elapsed_time = time.time() - start_time
print('CPU time to generate the scenario tree and blocks (s):', elapsed_time)

# Decomposition Parameters
ngo_rn_par = pymp.shared.dict()
ngo_th_par = pymp.shared.dict()
m.ngo_rn_par = Param(m.rn_r, m.n_stage, default=0, initialize=0, mutable=True)
m.ngo_th_par = Param(m.th_r, m.n_stage, default=0, initialize=0, mutable=True)
ngo_rn_par_k = pymp.shared.dict()
ngo_th_par_k = pymp.shared.dict()
cost_t = pymp.shared.dict()

m.mltp_o_rn = Param(m.rn_r, m.n_stage, m.iter, default=0, initialize=0, mutable=True)
m.mltp_o_th = Param(m.th_r, m.n_stage, m.iter, default=0, initialize=0, mutable=True)
m.cost = Param(m.n_stage, m.iter, default=0, initialize=0, mutable=True)

# Parameters to compute upper and lower bounds
m.cost_scenario = Param(n_stage[time_periods], m.iter, default=0, initialize=0, mutable=True)
m.mean = Param(m.iter, default=0, initialize=0, mutable=True)
m.std_dev = Param(m.iter, default=0, initialize=0, mutable=True)
m.cost_forward = Param(m.iter, default=0, initialize=0, mutable=True)
m.cost_UB = Param(m.iter, default=0, initialize=0, mutable=True)
m.cost_LB = Param(m.iter, default=0, initialize=0, mutable=True)
m.gap = Param(m.iter, default=0, initialize=0, mutable=True)
scenarios_iter = {}

# converting sets to lists:
rn_r = list(m.rn_r)
th_r = list(m.th_r)

# Retrieve duals
for t in m.t:
    for n in n_stage[t]:
        m.Bl[t, n].dual = Suffix(direction=Suffix.IMPORT)

# Add equality constraints
for t in m.t:
    for n in n_stage[t]:
        if t != 1:
            for (rn, r) in m.rn_r:
                for pn in n_stage[t - 1]:
                    if pn in parent_node[n]:
                        m.Bl[t, n].link_equal1.add(expr=(m.Bl[t, n].ngo_rn_prev[rn, r] ==
                                                         m.ngo_rn_par[rn, r, t - 1, pn]))
            for (th, r) in m.th_r:
                for pn in n_stage[t - 1]:
                    if pn in parent_node[n]:
                        m.Bl[t, n].link_equal2.add(expr=(m.Bl[t, n].ngo_th_prev[th, r] ==
                                                         m.ngo_th_par[th, r, t - 1, pn]))

# Stochastic Dual Dynamic integer Programming Algorithm (SDDiP)
for iter_ in m.iter:

    sample_index = deepcopy(random.sample(range(0, len(sc_nodes)), ns))
    sampled_scenarios = {}
    sampled_nodes = []
    sampled_nodes_stage = {}
    scenarios_iter[iter_] = {}
    for i in sample_index:
        n = sc_headers[i]
        sampled_scenarios[n] = sc_nodes[n]
        scenarios_iter[iter_][n] = list(sampled_scenarios[n])
    for t in m.t:
        for n in list(sampled_scenarios.keys()):
            s_sc = sampled_scenarios[n][len(m.t) - t]
            if s_sc not in sampled_nodes:
                sampled_nodes.append(s_sc)
            sampled_nodes_stage[t] = sampled_nodes
        sampled_nodes = []

    # Forward Pass
    for t in m.t:
        with pymp.Parallel(4) as p:
            for n in p.iterate(sampled_nodes_stage[t]):
                print("Time period", t)
                print("Current Node", n)

                if t != 1:
                    for (rn, r) in m.rn_r:
                        m.ngo_rn_par[rn, r, t-1, parent_node[n]] = ngo_rn_par[rn, r, t-1, parent_node[n]]
                        # print(rn, r, t-1, parent_node[n], m.ngo_rn_par[rn, r, t-1, parent_node[n]].value)
                    for (th, r) in m.th_r:
                        m.ngo_th_par[th, r, t-1,parent_node[n]] = ngo_th_par[th, r, t-1, parent_node[n]]
                    #     print(th, r, t-1, parent_node[n], m.ngo_th_par[th, r, t-1, parent_node[n]].value)
                    # print(t, cost_t[t-1, parent_node[n], iter_])

                # Solve the model
                mipsolver = SolverFactory('gurobi')
                mipsolver.options['mipgap'] = 0.0001
                mipsolver.options['timelimit'] = 40
                mipsolver.options['threads'] = 6
                mipsolver.solve(m.Bl[t, n])  # , tee=True)#,save_results=False)

                # Fix the linking variable as parameter for next t
                if t != m.t.last():
                    for (rn, r) in m.rn_r:
                        ngo_rn_par_k[rn, r, t, n, iter_] = m.Bl[t, n].ngo_rn[rn, r].value
                        ngo_rn_par[rn, r, t, n] = m.Bl[t, n].ngo_rn[rn, r].value

                    for (th, r) in m.th_r:
                        ngo_th_par_k[th, r, t, n, iter_] = m.Bl[t, n].ngo_th[th, r].value
                        ngo_th_par[th, r, t, n] = m.Bl[t, n].ngo_th[th, r].value

                # Store obj value to compute UB
                cost_t[t, n, iter_] = m.Bl[t, n].obj() - m.Bl[t, n].alphafut.value
                print('cost', t, n, cost_t[t, n, iter_])

    # Compute cost per scenario solved
    for s_sc in list(sampled_scenarios.keys()):
        m.cost_scenario[s_sc, iter_] = 0
        for t in m.t:
            m.cost_scenario[s_sc, iter_] += cost_t[t, sampled_scenarios[s_sc][len(m.t) - t], iter_]
        # print(s_sc, m.cost_scenario[s_sc, iter_].value)

    # Compute statistical upper bound
    m.mean[iter_] = (1 / ns) * sum(m.cost_scenario[s_sc, iter_] for s_sc in list(sampled_scenarios.keys()))
    print('mean', m.mean[iter_].value)
    m.std_dev[iter_] = math.sqrt((1 / (ns - 1)) * sum((m.cost_scenario[s_sc, iter_].value - m.mean[iter_].value) ** 2
                                                      for s_sc in list(sampled_scenarios.keys())))
    print('std dev', m.std_dev[iter_].value)
    m.cost_forward[iter_] = m.mean[iter_] + z_alpha_2 * (m.std_dev[iter_] / math.sqrt(ns))
    m.cost_UB[iter_] = min(m.cost_forward[kk].value for kk in m.iter if kk <= iter_)
    m.cost_UB.pprint()
    elapsed_time = time.time() - start_time
    print("CPU Time (s)", elapsed_time)

    # Backward Pass
    m.k.add(iter_)

    for t in reversed(list(m.t)):
        for n in sampled_nodes_stage[t]:
            print("Time period", t)
            print("Current Node", n)
            if t == m.t.last():
                m.Bl[t, n].alphafut.fix(0)
            else:
                m.Bl[t, n].alphafut.unfix()

            # Solve the model
            opt = SolverFactory('gurobi')
            opt.options['relax_integrality'] = 1
            opt.options['threads'] = 6
            # opt.options['SolutionNumber']=0
            opt.solve(m.Bl[t, n])  # , tee=True)#, save_results=False)#
            if t == 1:
                break

            # Get Lagrange multiplier from linking equality
            for rn_r_index in range(len(m.rn_r)):
                i = rn_r[rn_r_index][0]
                j = rn_r[rn_r_index][1]
                m.mltp_o_rn[i, j, t , n, iter_] = - m.Bl[t, n].dual[
                    m.Bl[t, n].link_equal1[rn_r_index + 1]]

            for th_r_index in range(len(m.th_r)):
                i = th_r[th_r_index][0]
                j = th_r[th_r_index][1]
                m.mltp_o_th[i, j, t, n, iter_] = - m.Bl[t, n].dual[
                    m.Bl[t, n].link_equal2[th_r_index + 1]]
            #                m.mltp_o_th.pprint()

            # Get optimal value
            m.cost[t, n, iter_] = m.Bl[t, n].obj()

            for pn in sampled_nodes_stage[t-1]:
                # add Benders cut for current iteration
                m.Bl[t-1, pn].fut_cost.add(expr=(m.Bl[t-1, pn].alphafut >= (1 / len(sampled_nodes_stage[t])) *
                                              sum(m.cost[t, n_, iter_]
                                                  + sum(m.mltp_o_rn[rn, r, t, n_, iter_] *
                                                        (ngo_rn_par_k[rn, r, t-1, pn, iter_] -
                                                         m.Bl[t-1, pn].ngo_rn[rn, r]) for rn, r in m.rn_r)
                                                  + sum(m.mltp_o_th[th, r, t, n_, iter_] *
                                                        (ngo_th_par_k[th, r, t-1, pn, iter_] -
                                                         m.Bl[t-1, pn].ngo_th[th, r]) for th, r in m.th_r)
                                                  for n_ in sampled_nodes_stage[t])))
            # m.Bl[t, n].fut_cost.pprint()

    # Compute lower bound
    m.cost_LB[iter_] = m.Bl[1, 'O'].obj()
    m.cost_LB.pprint()
    # Compute optimality gap
    m.gap[iter_] = (m.cost_UB[iter_] - m.cost_LB[iter_]) / m.cost_UB[iter_] * 100
    print(m.gap[iter_].value)

    if m.gap[iter_].value <= opt_tol:
        break

    elapsed_time = time.time() - start_time
    print("CPU Time (s)", elapsed_time)

elapsed_time = time.time() - start_time

print("Upper Bound", m.cost_UB[iter_].value)
print("Lower Bound", m.cost_LB[iter_].value)
print("Optimality gap (%)", m.gap[iter_].value)
print("CPU Time (s)", elapsed_time)