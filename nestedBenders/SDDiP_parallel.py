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
import csv

from scenarioTree import create_scenario_tree
import readData
import optBlocks as b
from forward import forward_pass
from backward_SDDiP import backward_pass

# ######################################################################################################################
# USER-DEFINED PARAMS

# Define case-study
curPath = os.path.abspath(os.path.curdir)
filepath = os.path.join(curPath, 'GTEPdata_5years.db')
time_periods = 5
stages = range(1, time_periods + 1)
scenarios = ['L', 'M', 'H']
single_prob = {'L': 1 / 3, 'M': 1 / 3, 'H': 1 / 3}

# Define parameters of the decomposition
max_iter = 10
opt_tol = 1  # %
ns = 20  # Number of scenarios solved per Forward/Backward Pass
# NOTE: ns should be between 1 and len(n_stage[time_periods])
z_alpha_2 = 1.96  # 95% confidence level

# ######################################################################################################################

# create scenarios and input data
nodes, n_stage, parent_node, children_node, prob, sc_nodes = create_scenario_tree(stages, scenarios, single_prob)
sc_headers = list(sc_nodes.keys())
readData.read_data(filepath, stages, n_stage)

# create blocks
m = b.create_model(time_periods, max_iter, n_stage, nodes, prob)
print('finished generating the blocks, started counting solution time')
start_time = time.time()

# Decomposition Parameters
ngo_rn_par = pymp.shared.dict()
ngo_th_par = pymp.shared.dict()
cost_forward = pymp.shared.dict()
m.ngo_rn_par = Param(m.rn_r, m.n_stage, default=0, initialize=0, mutable=True)
m.ngo_th_par = Param(m.th_r, m.n_stage, default=0, initialize=0, mutable=True)
m.ngo_rn_par_k = Param(m.rn_r, m.n_stage, m.iter, default=0, initialize=0, mutable=True)
m.ngo_th_par_k = Param(m.th_r, m.n_stage, m.iter, default=0, initialize=0, mutable=True)
m.cost_t = Param(m.n_stage, m.iter, default=0, initialize=0, mutable=True)

mltp_o_rn = pymp.shared.dict()
mltp_o_th = pymp.shared.dict()
cost_backward = pymp.shared.dict()
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
                print("Current thread", p.thread_num)

                ngo_rn, ngo_th, cost = forward_pass(m.Bl[t, n], rn_r, th_r)

                for (rn, r) in rn_r:
                    ngo_rn_par[rn, r, t, n] = ngo_rn[rn, r]
                for (th, r) in th_r:
                    ngo_th_par[th, r, t, n] = ngo_th[th, r]
                cost_forward[t, n] = cost
                print('cost', cost_forward[t, n])

        for n in sampled_nodes_stage[t]:
            if t != time_periods:
                for (rn, r) in rn_r:
                    m.ngo_rn_par[rn, r, t, n] = ngo_rn_par[rn, r, t, n]
                    m.ngo_rn_par_k[rn, r, t, n, iter_] = ngo_rn_par[rn, r, t, n]
                for (th, r) in th_r:
                    m.ngo_th_par[th, r, t, n] = ngo_th_par[th, r, t, n]
                    m.ngo_th_par_k[th, r, t, n, iter_] = ngo_th_par[th, r, t, n]
            else:
                for (rn, r) in rn_r:
                    m.ngo_rn_par[rn, r, t, n] = ngo_rn_par[rn, r, t, n]
                for (th, r) in th_r:
                    m.ngo_th_par[th, r, t, n] = ngo_th_par[th, r, t, n]
            m.cost_t[t, n, iter_] = cost_forward[t, n]

    # Compute cost per scenario solved
    for s_sc in list(sampled_scenarios.keys()):
        m.cost_scenario[s_sc, iter_] = 0
        for t in m.t:
            m.cost_scenario[s_sc, iter_] += m.cost_t[t, sampled_scenarios[s_sc][len(m.t) - t], iter_]
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
        # with pymp.Parallel(1) as pp:
        # for n in pp.iterate(sampled_nodes_stage[t]):
        for n in sampled_nodes_stage[t]:
            print("Time period", t)
            print("Current Node", n)
            # print("Current thread", pp.thread_num)

            mltp_rn, mltp_th, cost = backward_pass(t, m.Bl[t, n], time_periods, rn_r, th_r)

            cost_backward[t, n] = cost
            print('cost', t, n, cost_backward[t, n])

            if t != 1:
                for (rn, r) in rn_r:
                    mltp_o_rn[rn, r, t, n] = mltp_rn[rn, r]
                for (th, r) in th_r:
                    mltp_o_th[th, r, t, n] = mltp_th[th, r]

        for n in sampled_nodes_stage[t]:
            m.cost[t, n, iter_] = cost_backward[t, n]
            if t != 1:
                for (rn, r) in rn_r:
                    m.mltp_o_rn[rn, r, t, n, iter_] = mltp_o_rn[rn, r, t, n]
                for (th, r) in th_r:
                    m.mltp_o_th[th, r, t, n, iter_] = mltp_o_th[th, r, t, n]
                for pn in sampled_nodes_stage[t - 1]:
                    # add Benders cut for current iteration
                    m.Bl[t - 1, pn].fut_cost.add(expr=(m.Bl[t - 1, pn].alphafut >= (1 / len(sampled_nodes_stage[t])) *
                                                       sum(m.cost[t, n_, iter_]
                                                           + sum(m.mltp_o_rn[rn, r, t, n_, iter_] *
                                                                 (m.ngo_rn_par_k[rn, r, t - 1, pn, iter_] -
                                                                  m.Bl[t - 1, pn].ngo_rn[rn, r]) for rn, r in m.rn_r)
                                                           + sum(m.mltp_o_th[th, r, t, n_, iter_] *
                                                                 (m.ngo_th_par_k[th, r, t - 1, pn, iter_] -
                                                                  m.Bl[t - 1, pn].ngo_th[th, r]) for th, r in m.th_r)
                                                           for n_ in sampled_nodes_stage[t])))
            # m.Bl[t, n].fut_cost.pprint()

    # Compute lower bound
    m.cost_LB[iter_] = m.cost[1, 'O', iter_].value
    m.cost_LB.pprint()
    # Compute optimality gap
    m.gap[iter_] = (m.cost_UB[iter_] - m.cost_LB[iter_]) / m.cost_UB[iter_] * 100
    print(m.gap[iter_].value)

    if m.gap[iter_].value <= opt_tol:
        with open('results.csv', mode='w') as results_file:
            results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for t in m.t:
                for n in sampled_nodes_stage[t]:
                    for (rn, r) in rn_r:
                        if m.ngo_rn_par[rn, r, t, n].value != 0:
                            results_writer.writerow([rn, r, t, n, m.ngo_rn_par[rn, r, t, n].value])
                    for (th, r) in th_r:
                        if m.ngo_th_par[th, r, t, n].value != 0:
                            results_writer.writerow([th, r, t, n, m.ngo_th_par[th, r, t, n].value])
        break

    elapsed_time = time.time() - start_time
    print("CPU Time (s)", elapsed_time)

elapsed_time = time.time() - start_time

print("Upper Bound", m.cost_UB[iter_].value)
print("Lower Bound", m.cost_LB[iter_].value)
print("Optimality gap (%)", m.gap[iter_].value)
print("CPU Time (s)", elapsed_time)
