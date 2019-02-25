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
import csv

from scenarioTree import create_scenario_tree
import readData
import optBlocks as b
from forward import forward_pass
from backward_SDDiP import backward_pass

# ######################################################################################################################
# USER-DEFINED PARAMS

curPath = os.path.abspath(os.path.curdir)
filepath = os.path.join(curPath, 'data/GTEPdata_5years.db')
print(filepath)
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
ns = 30  # Number of scenarios solved per Forward/Backward Pass
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
m.cost_t = Param(m.n_stage, m.iter, default=0, initialize=0, mutable=True)

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
for stage in m.stages:
    for n in n_stage[stage]:
        if stage != 1:
            t_prev = t_per_stage[stage - 1][-1]
            for (rn, r) in m.rn_r:
                for pn in n_stage[stage - 1]:
                    if pn in parent_node[n]:
                        m.Bl[stage, n].link_equal1.add(expr=(m.Bl[stage, n].ngo_rn_prev[rn, r] ==
                                                         m.ngo_rn_par[rn, r, t_prev, pn]))
            for (th, r) in m.th_r:
                for pn in n_stage[stage - 1]:
                    if pn in parent_node[n]:
                        m.Bl[stage, n].link_equal2.add(expr=(m.Bl[stage, n].ngo_th_prev[th, r] ==
                                                         m.ngo_th_par[th, r, t_prev, pn]))

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
    for stage in m.stages:
        for n in sampled_nodes_stage[stage]:
            print("Stage", stage)
            print("Current Node", n)

            # for value stochastic (delete this later)
            # if t == 1:
            #     for (rn, r) in rn_r:
            #         if rn == 'ng-cc-new' and r == 'Coastal':
            #             m.Bl[t, n].ngb_rn[rn, r].fix(1)
            #         elif rn == 'ng-ct-new' and r == 'Northeast':
            #             m.Bl[t, n].ngb_rn[rn, r].fix(3)
            #         else:
            #             m.Bl[t, n].ngb_rn[rn, r].fix(0)

            ngo_rn, ngo_th, cost = forward_pass(m.Bl[stage, n], rn_r, th_r, t_per_stage[stage])
            # print(ngo_rn)

            for t in t_per_stage[stage]:
                for (rn, r) in rn_r:
                    m.ngo_rn_par[rn, r, t, n] = ngo_rn[rn, r, t]
                    m.ngo_rn_par_k[rn, r, t, n, iter_] = ngo_rn[rn, r, t]
                for (th, r) in th_r:
                    m.ngo_th_par[th, r, t, n] = ngo_th[th, r, t]
                    m.ngo_th_par_k[th, r, t, n, iter_] = ngo_th[th, r, t]
            m.cost_t[stage, n, iter_] = cost
            print('cost', m.cost_t[stage, n, iter_].value)

    # Compute cost per scenario solved
    for s_sc in list(sampled_scenarios.keys()):
        m.cost_scenario[s_sc, iter_] = 0
        for stage in m.stages:
            m.cost_scenario[s_sc, iter_] += m.cost_t[stage, sampled_scenarios[s_sc][len(m.stages) - stage], iter_]
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
    # m.k.add(iter_)

    for stage in reversed(list(m.stages)):
        for n in sampled_nodes_stage[stage]:
            print("Stage", stage)
            print("Current Node", n)

            # for value stochastic
            # if t == 1:
            #     for (rn, r) in rn_r:
            #         if rn == 'ng-cc-new' and r == 'Coastal':
            #             m.Bl[t, n].ngb_rn[rn, r].fix(1)
            #         elif rn == 'ng-ct-new' and r == 'Northeast':
            #             m.Bl[t, n].ngb_rn[rn, r].fix(3)
            #         else:
            #             m.Bl[t, n].ngb_rn[rn, r].fix(0)

            mltp_rn, mltp_th, cost = backward_pass(stage, m.Bl[stage, n], n_stages, rn_r, th_r)

            m.cost[stage, n, iter_] = cost
            print('cost', stage, n, m.cost[stage, n, iter_].value)

            if stage != 1:
                for (rn, r) in rn_r:
                    m.mltp_o_rn[rn, r, stage, n, iter_] = mltp_rn[rn, r]
                for (th, r) in th_r:
                    m.mltp_o_th[th, r, stage, n, iter_] = mltp_th[th, r]
                for pn in sampled_nodes_stage[stage-1]:
                    # add Benders cut for current iteration
                    t_prev = t_per_stage[stage - 1][-1]
                    m.Bl[stage-1, pn].fut_cost.add(expr=(m.Bl[stage-1, pn].alphafut >= (1 / len(sampled_nodes_stage[stage])) *
                                                  sum(m.cost[stage, n_, iter_]
                                                      + sum(m.mltp_o_rn[rn, r, stage, n_, iter_] *
                                                            (m.ngo_rn_par_k[rn, r, t_prev, pn, iter_] -
                                                             m.Bl[stage-1, pn].ngo_rn[rn, r, t_prev]) for rn, r in m.rn_r)
                                                      + sum(m.mltp_o_th[th, r, stage, n_, iter_] *
                                                            (m.ngo_th_par_k[th, r, t_prev, pn, iter_] -
                                                             m.Bl[stage-1, pn].ngo_th[th, r, t_prev]) for th, r in m.th_r)
                                                      for n_ in sampled_nodes_stage[stage])))
            # m.Bl[t, n].fut_cost.pprint()

    # Compute lower bound
    m.cost_LB[iter_] = m.cost[1, 'O', iter_].value
    m.cost_LB.pprint()
    # Compute optimality gap
    m.gap[iter_] = (m.cost_UB[iter_] - m.cost_LB[iter_]) / m.cost_UB[iter_] * 100
    print(m.gap[iter_].value)

    if m.gap[iter_].value <= opt_tol:
        # write results for state variables in a csv file
        with open('results.csv', mode='w') as results_file:
            results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for stage in m.stages:
                for n in sampled_nodes_stage[stage]:
                    for t in t_per_stage[stage]:
                        for (rn, r) in rn_r:
                            if m.ngo_rn_par[rn, r, t, n].value != 0:
                                results_writer.writerow([rn, r, t, n, m.ngo_rn_par[rn, r, t, n].value])
                        for (th, r) in th_r:
                            if m.ngo_th_par[th, r, t, n].value != 0:
                                results_writer.writerow([th, r, t, n, m.ngo_th_par[th, r, t, n].value])
        last_iter = iter_
        break

    elapsed_time = time.time() - start_time
    print("CPU Time (s)", elapsed_time)

elapsed_time = time.time() - start_time

print("Upper Bound", m.cost_UB[last_iter].value)
print("Lower Bound", m.cost_LB[last_iter].value)
print("Optimality gap (%)", m.gap[last_iter].value)
print("CPU Time (s)", elapsed_time)