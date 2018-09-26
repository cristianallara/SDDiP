__author__ = "Cristiana L. Lara"
# Nested Decomposition description at:
# http://www.optimization-online.org/DB_FILE/2017/08/6162.pdf

import time
import math
import random
from copy import deepcopy

from scenarioTree import create_scenario_tree
import readData
import optBlocks as b
from pyomo.environ import *

start_time = time.time()

# ######################################################################################################################
# USER-DEFINED PARAMS

# Define case-study
filepath = "C:\\Users\\cristianal\\PycharmProjects\\SDDP\\nestedBenders\\GTEPdata_5years.db"
time_periods = 5
stages = range(1, time_periods + 1)
scenarios = ['L', 'R', 'H']
single_prob = {'L': 0.25, 'R': 0.5, 'H': 0.25}

# Define parameters of the decomposition
max_iter = 2
opt_tol = 1  # %
ns = 5  # Number of scenarios solved per Forward/Backward Pass
# NOTE: ns should be between 1 and len(n_stage[time_periods])
z_alpha_2 = 1.96  # 95% confidence level

# ######################################################################################################################

# create scenarios and input data
nodes, n_stage, parent_node, prob, sc_nodes = create_scenario_tree(stages, scenarios, single_prob)
sc_headers = list(sc_nodes.keys())
readData.read_data(filepath, stages, n_stage)

# create blocks
m = b.create_model(time_periods, max_iter, n_stage, nodes, prob)
print('finished generating the blocks')
elapsed_time = time.time() - start_time
print('CPU time to generate the scenario tree and blocks (s):', elapsed_time)

# Parameters to compute upper and lower bounds
m.cost_scenario = Param(n_stage[time_periods], m.iter, default=0, initialize=0, mutable=True)
m.mean = Param(m.iter, default=0, initialize=0, mutable=True)
m.std_dev = Param(m.iter, default=0, initialize=0, mutable=True)
m.cost_forward = Param(m.iter, default=0, initialize=0, mutable=True)
m.cost_UB = Param(m.iter, default=0, initialize=0, mutable=True)
m.cost_LB = Param(m.iter, default=0, initialize=0, mutable=True)
m.gap = Param(m.iter, default=0, initialize=0, mutable=True)
scenarios_iter = {}

# Retrieve duals
for t in m.t:
    for n in n_stage[t]:
        m.Bl[t, n].dual = Suffix(direction=Suffix.IMPORT)

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
    print(sampled_scenarios)
    for t in m.t:
        for n in list(sampled_scenarios.keys()):
            k = sampled_scenarios[n][len(m.t) - t]
            if k not in sampled_nodes:
                sampled_nodes.append(k)
            sampled_nodes_stage[t] = sampled_nodes
        sampled_nodes = []

    # Forward Pass

    for t in m.t:
        for n in sampled_nodes_stage[t]:
            print("Time period", t)
            print("Current Node", n)

            # Fix alphafut=0 for iter 1
            if iter_ == 1:
                m.Bl[t, n].alphafut.fix(0)

            # add equality linking with parental nodes for stages =/= 1
            if iter_ == 1:
                if t != 1:
                    for (rn, r) in m.rn_r:
                        for pn in n_stage[t - 1]:
                            if pn in parent_node[n]:
                                m.Bl[t, n].link_equal1.add(expr=(m.Bl[t, n].ngo_rn_prev[rn, r] ==
                                                                 m.ngo_rn_par[rn, r, t - 1, pn]))
                                if t > readData.LT[rn]:
                                    m.Bl[t, n].link_equal3.add(expr=(m.Bl[t, n].ngb_rn_LT[rn, r] ==
                                                                     m.ngb_rn_par[rn, r, t - m.LT[rn], pn]))
                    for (th, r) in m.th_r:
                        for pn in n_stage[t - 1]:
                            if pn in parent_node[n]:
                                m.Bl[t, n].link_equal2.add(expr=(m.Bl[t, n].ngo_th_prev[th, r] ==
                                                                 m.ngo_th_par[th, r, t - 1, pn]))
                                if t > readData.LT[th]:
                                    m.Bl[t, n].link_equal4.add(expr=(m.Bl[t, n].ngb_th_LT[th, r] ==
                                                                     m.ngo_th_par[th, r, t - m.LT[th], pn]))

            # Solve the model
            mipsolver = SolverFactory('gurobi')  # _persistent')
            #                mipsolver.set_instance(m.Bl[t,n,pn])
            mipsolver.options['mipgap'] = 0.0001
            mipsolver.options['timelimit'] = 40
            mipsolver.options['threads'] = 6
            results = mipsolver.solve(m.Bl[t, n])  # , tee=True)#,save_results=False)

            # Fix the linking variable as parameter for next t
            if t != m.t.last():
                for (rn, r) in m.rn_r:
                    m.ngo_rn_par[rn, r, t, n] = m.Bl[t, n].ngo_rn[rn, r].value
                    m.ngb_rn_par[rn, r, t, n] = m.Bl[t, n].ngb_rn[rn, r].value
                for (th, r) in m.th_r:
                    m.ngo_th_par[th, r, t, n] = m.Bl[t, n].ngo_th[th, r].value
                    m.ngb_th_par[th, r, t, n] = m.Bl[t, n].ngb_th[th, r].value

            for (rn, r) in m.rn_r:
                m.ngo_rn_par_k[rn, r, t, n, iter_] = m.Bl[t, n].ngo_rn[rn, r].value
                m.ngb_rn_par_k[rn, r, t, n, iter_] = m.Bl[t, n].ngb_rn[rn, r].value

            for (th, r) in m.th_r:
                m.ngo_th_par_k[th, r, t, n, iter_] = m.Bl[t, n].ngo_th[th, r].value
                m.ngb_th_par_k[th, r, t, n, iter_] = m.Bl[t, n].ngb_th[th, r].value

            # Store obj value to compute UB
            m.cost_t[t, n, iter_] = m.Bl[t, n].obj() - m.Bl[t, n].alphafut.value
            # print('cost', m.cost_t[t, n, iter_].value)

    # Compute cost per scenario solved
    for k in list(sampled_scenarios.keys()):
        m.cost_scenario[k, iter_] = 0
        for t in m.t:
            m.cost_scenario[k, iter_] += m.cost_t[t, sampled_scenarios[k][len(m.t) - t], iter_]
        # print(k, m.cost_scenario[k, iter_].value)

    # Compute statistical upper bound
    m.mean[iter_] = (1 / ns) * sum(m.cost_scenario[k, iter_] for k in list(sampled_scenarios.keys()))
    print('mean', m.mean[iter_].value)
    m.std_dev[iter_] = math.sqrt((1 / (ns - 1)) * sum((m.cost_scenario[k, iter_] - m.mean[iter_]) ** 2
                                                      for k in list(sampled_scenarios.keys())))
    print('std dev', m.std_dev[iter_].value)
    m.cost_forward[iter_] = m.mean[iter_] + z_alpha_2 * (m.std_dev[iter_] / math.sqrt(ns))
    m.cost_UB[iter_] = min(m.cost_forward[kk].value for kk in m.iter if kk <= iter_)
    m.cost_UB.pprint()
    m.Bl[t, n].alphafut.unfix()
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

            # add Benders cut
            if t != m.t.last():
                for k in m.k:
                    # for pn_ in N_stage[t]:
                    #     if pn_ == n:
                    m.Bl[t, n].fut_cost.add(expr=(m.Bl[t, n].alphafut >=
                                                  sum((m.prob[n_] / m.prob[n]) * m.cost[t + 1, n_, k]
                                                      for n_ in n_stage[t + 1] if n in parent_node[n_])
                                                  + sum((m.prob[n_] / m.prob[n]) *
                                                        m.mltp_o_rn[rn, r, t + 1, n_, k] *
                                                        (m.ngo_rn_par_k[rn, r, t, n, k] -
                                                         m.Bl[t, n].ngo_rn[rn, r]) for rn, r in m.rn_r
                                                        for n_ in n_stage[t + 1] if n in parent_node[n_])
                                                  + sum((m.prob[n_] / m.prob[n]) *
                                                        m.mltp_o_th[th, r, t + 1, n_, k] *
                                                        (m.ngo_th_par_k[th, r, t, n, k] -
                                                         m.Bl[t, n].ngo_th[th, r]) for th, r in m.th_r
                                                        for n_ in n_stage[t + 1] if n in parent_node[n_])
                                                  + sum((m.prob[n_] / m.prob[n]) *
                                                        m.mltp_b_rn[rn, r, t + m.LT[rn], n_, k] *
                                                        (m.ngb_rn_par_k[rn, r, t, n, k] -
                                                         m.Bl[t, n].ngb_rn[rn, r]) for rn, r in m.rn_r
                                                        for n_ in n_stage[t + 1] if n in parent_node[n_]
                                                        and (t + m.LT[rn] <= m.t.last()))
                                                  + sum((m.prob[n_] / m.prob[n]) *
                                                        m.mltp_b_th[th, r, t + m.LT[th], n_, k] *
                                                        (m.ngb_th_par_k[th, r, t, n, k] -
                                                         m.Bl[t, n].ngb_th[th, r]) for th, r in m.th_r
                                                        for n_ in n_stage[t + 1] if n in parent_node[n_]
                                                        and (t + m.LT[th] <= m.t.last()))))
            #        m.Bl[t,n,pn].fut_cost.pprint()

            # Solve the model
            # opt = SolverFactory('cplex')
            opt = SolverFactory('gurobi')
            # opt.set_instance(m.Bl[t])
            opt.options['relax_integrality'] = 1
            opt.options['threads'] = 6
            # opt.options['SolutionNumber']=0
            results = opt.solve(m.Bl[t, n])  # , tee=True)#, save_results=False)#
            #                m.Bl[t,n,pn].alphafut.pprint()

            # Get Lagrange multiplier from linking equality
            if t != m.t.first():
                for rn_r_index in range(len(m.rn_r)):
                    i = m.rn_r[rn_r_index][0]
                    j = m.rn_r[rn_r_index][1]
                    m.mltp_o_rn[i, j, t, n, iter_] = - m.Bl[t, n].dual[m.Bl[t, n].link_equal1[rn_r_index + 1]]
                    if t > m.LT[rn]:
                        m.mltp_b_rn[i, j, t, n, iter_] = - m.Bl[t, n].dual[m.Bl[t, n].link_equal3[rn_r_index + 1]]
                    else:
                        m.mltp_b_rn[i, j, t, n, iter_] = 0

            if t != m.t.first():
                for th_r_index in range(len(m.th_r)):
                    i = m.th_r[th_r_index][0]
                    j = m.th_r[th_r_index][1]
                    m.mltp_o_th[i, j, t, n, iter_] = - m.Bl[t, n].dual[m.Bl[t, n].link_equal2[th_r_index + 1]]
                    if t > m.LT[th]:
                        m.mltp_b_th[i, j, t, n, iter_] = - m.Bl[t, n].dual[m.Bl[t, n].link_equal4[th_r_index + 1]]
                    else:
                        m.mltp_b_th[i, j, t, n, iter_] = 0
            #                m.mltp_o_th.pprint()

            # Get optimal value
            m.cost[t, n, iter_] = m.Bl[t, n].obj()

    # Compute lower bound
    m.cost_LB[iter_] = m.cost[1, 'O', iter_]
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
