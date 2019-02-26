__author__ = "Cristiana L. Lara"
# Stochastic Nested Decomposition description at:
# https://link.springer.com/content/pdf/10.1007%2Fs10107-018-1249-5.pdf

import time
import math
import random
from copy import deepcopy
from pyomo.environ import *
import os.path

from scenarioTree import create_scenario_tree
import readData
import optBlocks as b

# ######################################################################################################################
# USER-DEFINED PARAMS

# Define case-study
curPath = os.path.abspath(os.path.curdir)
filepath = os.path.join(curPath, 'data/GTEPdata_5years.db')
n_stages = 5  # number od stages in the scenario tree
stages = range(1, n_stages + 1)
scenarios = ['L', 'M', 'H']
single_prob = {'L': 1 / 3, 'M': 1 / 3, 'H': 1 / 3}

time_periods = 5
set_time_periods = range(1, time_periods + 1)
# t_per_stage = {1: [1, 2], 2: [3, 4], 3: [5, 6], 4: [7, 8], 5: [9, 10]}
t_per_stage = {1: [1], 2: [2], 3: [3], 4: [4], 5: [5]}

# Define parameters of the decomposition
max_iter = 50
opt_tol = 1  # %
ns = 81  # Number of scenarios solved per Forward/Backward Pass
# NOTE: ns should be between 1 and len(n_stage[time_periods])
z_alpha_2 = 1.96  # 95% confidence level

# ######################################################################################################################

# create scenarios and input data
nodes, n_stage, parent_node, children_node, prob, sc_nodes = create_scenario_tree(stages, scenarios, single_prob)
sc_headers = list(sc_nodes.keys())
readData.read_data(filepath, stages, n_stage, t_per_stage)

# create blocks
m = b.create_model(n_stages, time_periods, t_per_stage, max_iter, n_stage, nodes, readData.L_max_s, 0)
print('finished generating the blocks, started counting solution time')
start_time = time.time()

# Decomposition Parameters
m.ngo_rn_par = Param(m.rn_r, m.n_stage, default=0, initialize=0, mutable=True)
m.ngo_th_par = Param(m.th_r, m.n_stage, default=0, initialize=0, mutable=True)
m.ngo_rn_par_k = Param(m.rn_r, m.n_stage, m.iter, default=0, initialize=0, mutable=True)
m.ngo_th_par_k = Param(m.th_r, m.n_stage, m.iter, default=0, initialize=0, mutable=True)
m.cost = Param(m.n_stage, m.iter, default=0, initialize=0, mutable=True)
m.mltp_o_rn = Param(m.rn_r, m.n_stage, m.iter, default=0, initialize=0, mutable=True)
m.mltp_o_th = Param(m.th_r, m.n_stage, m.iter, default=0, initialize=0, mutable=True)
m.cost_stage = Param(m.n_stage, m.iter, default=0, initialize=0, mutable=True)

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

expl_nodes = {}
for iter_ in m.iter:
    for n in nodes:
        expl_nodes[n, iter_] = 0

# Stochastic Nested Benders Decomposition (SND)
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
    for stage in m.stages:
        for n in sampled_nodes_stage[stage]:
            print("Stage", stage)
            print("Current Node", n)

            # Solve the model
            mipsolver = SolverFactory('gurobi')
            mipsolver.options['mipgap'] = 0.0001
            mipsolver.options['timelimit'] = 40
            mipsolver.options['threads'] = 6
            mipsolver.solve(m.Bl[stage, n])  # , tee=True)#,save_results=False)

            expl_nodes[n, iter_] = 1

            # Fix the linking variable as parameter for next t
            if stage != m.stages.last():
                for t in t_per_stage[stage]:
                    for (rn, r) in m.rn_r:
                        m.ngo_rn_par_k[rn, r, t, n, iter_] = m.Bl[stage, n].ngo_rn[rn, r, t].value
                        m.ngo_rn_par[rn, r, t, n] = m.Bl[stage, n].ngo_rn[rn, r, t].value

                    for (th, r) in m.th_r:
                        m.ngo_th_par_k[th, r, t, n, iter_] = m.Bl[stage, n].ngo_th[th, r, t].value
                        m.ngo_th_par[th, r, t, n] = m.Bl[stage, n].ngo_th[th, r, t].value

            # Store obj value to compute UB
            m.cost_stage[stage, n, iter_] = m.Bl[stage, n].obj() - m.Bl[stage, n].alphafut.value
            print('cost', m.cost_stage[stage, n, iter_].value)

    # Compute cost per scenario solved
    for s_sc in list(sampled_scenarios.keys()):
        m.cost_scenario[s_sc, iter_] = 0
        for stage in m.stages:
            m.cost_scenario[s_sc, iter_] += m.cost_stage[stage, sampled_scenarios[s_sc][len(m.stages) - stage], iter_]
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

    for stage in reversed(list(m.stages)):
        if stage != m.stages.last():
            for n in n_stage[stage]:
                print("Stage", stage)
                print("Current Node", n)
                if n in sampled_nodes_stage[stage]:
                    for cn in children_node[n]:
                        print("Children Node", cn, "of stage", stage + 1)
                        if cn in n_stage[m.stages.last()]:
                            m.Bl[stage + 1, cn].alphafut.fix(0)
                        else:
                            m.Bl[stage + 1, cn].alphafut.unfix()

                        # Solve the model
                        opt = SolverFactory('gurobi')
                        opt.options['relax_integrality'] = 1
                        opt.options['threads'] = 6
                        # opt.options['SolutionNumber']=0
                        opt.solve(m.Bl[stage + 1, cn])  # , tee=True)#, save_results=False)#

                        # Get Lagrange multiplier from linking equality
                        for rn_r_index in range(len(m.rn_r)):
                            i = rn_r[rn_r_index][0]
                            j = rn_r[rn_r_index][1]
                            m.mltp_o_rn[i, j, stage + 1, cn, iter_] = - m.Bl[stage + 1, cn].dual[
                                m.Bl[stage + 1, cn].link_equal1[rn_r_index + 1]]

                        for th_r_index in range(len(m.th_r)):
                            i = th_r[th_r_index][0]
                            j = th_r[th_r_index][1]
                            m.mltp_o_th[i, j, stage + 1, cn, iter_] = - m.Bl[stage + 1, cn].dual[
                                m.Bl[stage + 1, cn].link_equal2[th_r_index + 1]]
                        #                m.mltp_o_th.pprint()

                        # Get optimal value
                        m.cost[stage + 1, cn, iter_] = m.Bl[stage + 1, cn].obj()

                    # add Benders cut for current iteration
                    t = t_per_stage[stage][-1]
                    m.Bl[stage, n].fut_cost.add(expr=(m.Bl[stage, n].alphafut >=
                                                  sum((prob[cn] / prob[n]) * m.cost[stage + 1, cn, iter_]
                                                      + sum((prob[cn] / prob[n]) *
                                                            m.mltp_o_rn[rn, r, stage + 1, cn, iter_] *
                                                            (m.ngo_rn_par_k[rn, r, t, n, iter_] -
                                                             m.Bl[stage, n].ngo_rn[rn, r, t]) for rn, r in m.rn_r)
                                                      + sum((prob[cn] / prob[n]) *
                                                            m.mltp_o_th[th, r, stage + 1, cn, iter_] *
                                                            (m.ngo_th_par_k[th, r, t, n, iter_] -
                                                             m.Bl[stage, n].ngo_th[th, r, t]) for th, r in m.th_r)
                                                      for cn in children_node[n])))
                    # m.Bl[t, n].fut_cost.pprint()

    opt.solve(m.Bl[1, 'O'])
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

# post_process()
